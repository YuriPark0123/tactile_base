"""Run a force task (grasp_chip) seed by seed with a solver-speed watchdog.

The UIPC solve of grasp_chip is not deterministic across launches: with the same seed and code, some launches settle into
a state where every step needs the full Newton budget (~4-5 s/step, an episode takes ~40 min and exceeds the worker wall
time), while others run at ~0.1 s/step in pre-move and ~2 steps/s with the policy (~5 min per episode). The split is
visible within the first ~25 steps (fast: step 25 at ~36 s, slow: at ~110 s). This runner launches each seed as the
standard force-task worker (same environment as `dispatch_force_task_seeds`), kills a launch whose step `--probe_step`
is reached later than `--probe_seconds`, and relaunches the same seed (fresh scene, same seed) up to `--max_tries`.

Output: <out>/attempts/<seed>/{try<k>.log, result.json}, <out>/summary.json; the worker writes videos / trajectories /
dagger records into <out> as usual.

Usage (from the OpenViTac root):
  python scripts/run_force_seeds.py --out eval_result/openpi/grasp_chip/base20 --seeds 10000-10019 -- \
      grasp_chip task_config/grasp_chip.yml policy/openpi/abs_joint/deploy_openvtla_grasp_chip.yml \
      --eval_step_timeout_seconds 0 --headless
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STEP = re.compile(r"Step\s+(\d+)\(.*?Running\s+([\d.]+)s")


def kill_group(p: subprocess.Popen):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass


def wait_sim_exit(timeout: float = 180):
    """Wait until killed Isaac Sim workers have exited (a new launch next to a dying one is very slow)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run(["pgrep", "-f", "OPENVITAC_FORCE_WORKER|scripts/eval_policy.py"], capture_output=True, text=True)
        if not r.stdout.strip():
            return
        time.sleep(5)


def run_once(cmd, env, log: Path, probe_step: int, probe_seconds: float, wall: float) -> tuple[str, float | None]:
    with log.open("w") as stream:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    t0, probed, speed = time.time(), False, None
    try:
        while p.poll() is None:
            time.sleep(5)
            if time.time() - t0 > wall:
                kill_group(p)
                return "wall_timeout", speed
            if not probed:
                text = log.read_bytes()[-20000:].decode("utf-8", "ignore").replace("\r", "\n")
                steps = [(int(a), float(b)) for a, b in STEP.findall(text)]
                if steps:
                    n, running = steps[-1]
                    if n >= probe_step:
                        probed, speed = True, running
                        if running > probe_seconds:
                            kill_group(p)
                            return "slow", speed
                    elif running > probe_seconds:  # not even at the probe step yet
                        kill_group(p)
                        return "slow", running
        return ("ok" if p.returncode == 0 else f"exit_{p.returncode}"), speed
    finally:
        if p.poll() is None:
            kill_group(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", required=True, help="a-b inclusive")
    ap.add_argument("--max_tries", type=int, default=8)
    ap.add_argument("--probe_step", type=int, default=25)
    ap.add_argument("--probe_seconds", type=float, default=60.0)
    ap.add_argument("--wall", type=float, default=1800.0)
    ap.add_argument("eval_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    eval_args = [x for x in a.eval_args if x != "--"]
    out = (REPO / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s0, s1 = (int(x) for x in a.seeds.split("-"))
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"seeds": {}}
    for seed in range(s0, s1 + 1):
        if summary["seeds"].get(str(seed), {}).get("result") in ("success", "fail"):
            continue
        att = out / "attempts" / str(seed)
        att.mkdir(parents=True, exist_ok=True)
        result_path = att / "result.json"
        tries = []
        for k in range(a.max_tries):
            result_path.unlink(missing_ok=True)
            env = os.environ.copy()
            env.update(OPENVITAC_FORCE_WORKER="1", OPENVITAC_FORCE_OUTPUT=str(out), OPENVITAC_FORCE_RESULT=str(result_path),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HEADLESS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                       OPENVITAC_LIVESTREAM="0", OMNI_KIT_ACCEPT_EULA="yes")
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            cmd = [sys.executable, "-u", "-B", str(REPO / "scripts/eval_policy.py"), *eval_args,
                   "--start_seed", str(seed), "--max_seed", str(seed), "--total_num", "1"]
            t0 = time.time()
            status, speed = run_once(cmd, env, att / f"try{k}.log", a.probe_step, a.probe_seconds, a.wall)
            row = {"try": k, "status": status, "probe_running_s": speed, "wall_s": round(time.time() - t0, 1)}
            if status == "ok" and result_path.exists():
                row.update(json.loads(result_path.read_text()))
            tries.append(row)
            print(f"[force-runner] seed {seed} try {k}: {row}", flush=True)
            wait_sim_exit()
            if row.get("result") in ("success", "fail"):
                break
            time.sleep(10)
        final = tries[-1].get("result", "error")
        summary["seeds"][str(seed)] = {"result": final, "tries": tries}
        done = [v for v in summary["seeds"].values() if v["result"] in ("success", "fail")]
        summary.update(completed=len(done), successes=sum(v["result"] == "success" for v in done))
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"[force-runner] seed {seed}: {final}; {summary['successes']}/{summary['completed']}", flush=True)


if __name__ == "__main__":
    main()

"""Serve a trained WorldTactile pi0.5 run to a simulator adapter (wire format: worldtactile/serving/protocol.py).

  python scripts/serve.py --run /data/experiments/pi05_wt_delta_s2b --step stage2_final --port 6123 \
      [--query_log <dir>/query_log_insert_HDMI.jsonl] [--rtc 0 --rtc_blend 0] [--execute_steps 25]

Prints "READY on 127.0.0.1:<port>" once the model is loaded.
"""
import argparse
import sys
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldtactile.serving.policy import Policy  # noqa: E402
from worldtactile.serving.protocol import AUTHKEY, pack, unpack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--step", default="stage2_final",
                    help="checkpoint tag after 'trainable_' (e.g. stage2_final); a number N means stepN")
    ap.add_argument("--port", type=int, default=6123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--query_log", default=None, help="jsonl path for the closed-loop tactile prediction log")
    ap.add_argument("--rtc", type=int, default=1, help="real-time chunking guidance toward the previous chunk")
    ap.add_argument("--rtc_beta", type=float, default=20.0, help="RTC guidance weight clip")
    ap.add_argument("--rtc_blend", type=int, default=1, help="mix the overlap with the previous chunk (RTC weights)")
    ap.add_argument("--execute_steps", type=int, default=25, help="actions executed per chunk; must match the adapter")
    ap.add_argument("--smooth_window", type=int, default=0, help="odd Savitzky-Golay window for the arm (0 = off)")
    args = ap.parse_args()
    tag = f"step{args.step}" if args.step.isdigit() else args.step
    policy = Policy(args.run, tag, device=args.device, rtc=bool(args.rtc), rtc_beta=args.rtc_beta,
                    rtc_blend=bool(args.rtc_blend), execute_steps=args.execute_steps,
                    smooth_window=args.smooth_window, query_log=args.query_log)
    print(f"pi05 server: rtc={policy.rtc} beta={policy.rtc_beta} blend={policy.rtc_blend} "
          f"smooth_window={policy.smooth_window} execute_steps={policy.execute_steps}", flush=True)
    with Listener(("127.0.0.1", args.port), authkey=AUTHKEY) as listener:
        print(f"READY on 127.0.0.1:{args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        blob = conn.recv_bytes()
                    except EOFError:
                        break
                    actions = policy.act(unpack(blob))
                    conn.send_bytes(pack(actions=actions.astype(np.float32), target_offset=np.array(policy.target_offset)))


if __name__ == "__main__":
    main()

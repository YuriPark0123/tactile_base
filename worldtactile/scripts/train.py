"""Two-stage training of WorldTactile on pi0.5.

  python scripts/train.py --run_name pi05_wt --tasks insert_hole,insert_HDMI,put_bottle_in_shelf

Stage 1  trainable: tactile queries, tactile input projection + finger embedding, tactile prediction head.
         loss: align (queries -> future AnyTouch2 CLS). pi0.5 frozen.
Stage 2  trainable: stage 1 + PaliGemma LoRA (attention + MLP, all 18 layers) + the whole action expert and the
         action/time projections (fp32 master weights). SigLIP and the VLM base weights stay frozen.
         loss: flow + align_weight * align + adhere_weight * max(0, margin - (flow(corrupted tactile) - flow)).
A stage ends when it has converged (the median of the logged batch losses over the last `conv_window` steps improved by
less than `conv_thresh` against the window before, `conv_patience` checks in a row; stage 1: align, stage 2: flow and
align), or at its time cap. Stage 2 uses a cosine schedule over the steps that fit in the remaining budget.
Tasks are sampled equally often. Writes to <WT_EXP_ROOT>/<run_name>/: args.json, norm_stats.npz, cls_stats.npz,
trainable_step<N>.pt (last two kept), trainable_stage<k>_final.pt, optim_latest.pt.
"""
import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldtactile import paths  # noqa: E402
from worldtactile.dataset import UniVTACContactDataset  # noqa: E402
from worldtactile.pi05.collate import Pi05Collator, to_device  # noqa: E402
from worldtactile.pi05.model import WorldTactilePi05  # noqa: E402

TRAINABLE_PREFIXES = ("queries", "finger_emb", "tactile_in", "align_proj", "pi.action_", "pi.time_mlp",
                      "pi.paligemma_with_expert.gemma_expert")


def make_loader(args):
    tasks = args.tasks.split(",")
    ds = UniVTACContactDataset(tasks, horizon=args.horizon, cameras=args.cameras.split(","),
                               tactile_corrupt=args.adhere_weight > 0, target_offset=args.target_offset)
    counts = {t: len(ds.by_task[t]) for t in tasks}
    n_task = {t: sum(ds.episodes[i]["end"] - ds.episodes[i]["start"] + 1 for i in ds.by_task[t]) for t in tasks}
    w = [1.0 / n_task[ds.episodes[i]["task"]] for i, _ in ds.samples]  # every task drawn equally often
    sampler = WeightedRandomSampler(w, num_samples=10 ** 8, replacement=True)
    # workers return raw items (PIL images); image/prompt processing runs in the main process (small /dev/shm)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, collate_fn=lambda x: x,
                    persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
    return ds, dl, counts


def converged(hist, keys, window, thresh):
    if not hist:
        return False, {}
    last = hist[-1][0]
    cur = [h for h in hist if last - window < h[0] <= last]
    prev = [h for h in hist if last - 2 * window < h[0] <= last - window]
    if len(cur) < 10 or len(prev) < 10:
        return False, {}
    gain = {k: 1 - st.median(h[1][k] for h in cur) / st.median(h[1][k] for h in prev) for k in keys}
    return all(g < thresh for g in gain.values()), gain


def fp32_master(model):
    """Fully trained pretrained parts (action expert, action/time projections) keep fp32 weights."""
    for m in (model.expert, model.pi.action_in_proj, model.pi.action_out_proj, model.pi.time_mlp_in,
              model.pi.time_mlp_out):
        for p in m.parameters():
            p.data = p.data.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="insert_hole,insert_HDMI,put_bottle_in_shelf")
    ap.add_argument("--cameras", default="head,wrist")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--target_offset", type=int, default=24, help="future target clip ends at t + offset")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lr_new", type=float, default=1e-4, help="queries, tactile input, tactile head")
    ap.add_argument("--lr_lora", type=float, default=5e-5)
    ap.add_argument("--lr_expert", type=float, default=2.5e-5, help="pretrained action expert + projections")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--align_weight", type=float, default=1.0)
    ap.add_argument("--adhere_weight", type=float, default=0.1)
    ap.add_argument("--adhere_margin", type=float, default=5.0)
    ap.add_argument("--budget_hours", type=float, default=36.0)
    ap.add_argument("--stage1_max_hours", type=float, default=8.0)
    ap.add_argument("--conv_window", type=int, default=1000)
    ap.add_argument("--conv_thresh", type=float, default=0.05)
    ap.add_argument("--conv_patience", type=int, default=1,
                    help="consecutive converged checks needed to end a stage (guards against one noisy window)")
    ap.add_argument("--min_steps1", type=int, default=1500)
    ap.add_argument("--min_steps2", type=int, default=3000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--measure_memory", action="store_true", help="run a few stage-2 steps, print memory, exit")
    ap.add_argument("--resume", default="", help="trainable_*.pt to start from (with --start_stage / --start_step)")
    ap.add_argument("--start_stage", type=int, default=1)
    ap.add_argument("--start_step", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    out = paths.EXP_ROOT / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    ds, dl, counts = make_loader(args)
    col = Pi05Collator(args.cameras.split(","))
    model = WorldTactilePi05(str(paths.PI05_BASE), horizon=args.horizon, gradient_checkpointing=True)
    model.add_lora(args.lora_rank)
    for p in model.new_module_params():
        p.data = p.data.float()
    if args.resume:
        if args.start_stage >= 2:  # stage-2 weights were saved from fp32 master copies; keep their precision
            fp32_master(model)
        missing, unexpected = model.load_state_dict(torch.load(args.resume, map_location="cpu"), strict=False)
        assert not unexpected, unexpected[:5]
        print(f"resumed weights from {args.resume} (stage {args.start_stage}, step {args.start_step})", flush=True)
    model.to(dev).train()
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    np.savez(out / "norm_stats.npz", **ds.stats)
    np.savez(out / "cls_stats.npz", mean=ds.cls_mean, std=ds.cls_std)
    n_lora = sum(p.numel() for n, p in model.lm.named_parameters() if "lora_" in n)
    print(f"tasks {counts} | samples {len(ds)} | LoRA rank {args.lora_rank}: {n_lora / 1e6:.1f}M | "
          f"expert {sum(p.numel() for p in model.expert.parameters()) / 1e6:.1f}M", flush=True)

    holder = {}

    def save(tag):
        keep = {n: p.detach().cpu() for n, p in model.named_parameters()
                if "lora_" in n or n.startswith(TRAINABLE_PREFIXES)}
        torch.save(keep, out / f"trainable_{tag}.pt")
        if "opt" in holder:
            torch.save(holder["opt"].state_dict(), out / "optim_latest.pt")

    it = iter(dl)
    t_start = time.time()
    step = args.start_step
    stages = [2] if args.measure_memory else [s for s in (1, 2) if s >= args.start_stage]
    for stage in stages:
        params = model.set_stage(stage)
        if stage == 2:
            fp32_master(model)
        new_ids = {id(p) for p in model.new_module_params()}
        groups = [dict(params=[p for p in params if id(p) in new_ids], lr=args.lr_new)]
        if stage == 2:
            groups.append(dict(params=[p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n],
                               lr=args.lr_lora))
            rest = {id(p) for g in groups for p in g["params"]}
            groups.append(dict(params=[p for p in params if id(p) not in rest], lr=args.lr_expert))
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-4)
        holder["opt"] = opt
        base = [g["lr"] for g in opt.param_groups]
        n_train = sum(p.numel() for p in params)
        stage_start, s0, planned, hist, streak = time.time(), step, None, [], 0
        cap_h = args.stage1_max_hours if stage == 1 else args.budget_hours
        print(f"=== stage {stage}: {n_train / 1e6:.1f}M trainable params ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        while True:
            k = step - s0
            if stage == 2 and planned is None and k == 50:
                sps = (time.time() - stage_start) / 50
                planned = 50 + int((args.budget_hours * 3600 - (time.time() - t_start)) / sps)
                print(f"stage 2: {sps:.2f} s/step, cosine schedule over {planned} steps", flush=True)
            if k < args.warmup:
                scale = (k + 1) / args.warmup
            elif stage == 2 and planned:
                scale = 0.5 * (1 + math.cos(math.pi * min(1.0, (k - args.warmup) / max(1, planned - args.warmup))))
            else:
                scale = 1.0
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * scale
            batch = to_device(col(next(it)), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logs = model(batch, stage=stage, align_weight=args.align_weight,
                                   adhere_weight=args.adhere_weight, adhere_margin=args.adhere_margin)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            elapsed = (time.time() - t_start) / 3600
            if args.measure_memory and step - s0 >= 4:
                print(f"MEASURE rank {args.lora_rank} batch {args.batch}: peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                      f"{(time.time() - stage_start) / (step - s0):.2f} s/step", flush=True)
                return
            if step % args.log_every == 0:
                vals = {kk: round(float(v), 5) for kk, v in logs.items()}
                hist.append((step - s0, {kk: float(v) for kk, v in logs.items()}))
                print(f"step {step} stage {stage} {vals} gnorm {float(gn):.3f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{(time.time() - stage_start) / (step - s0):.2f}s/step "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB elapsed {elapsed:.2f}h", flush=True)
            if step % args.save_every == 0:
                save(f"step{step}")
                for old in sorted(out.glob("trainable_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))[:-2]:
                    old.unlink()
                if step - s0 >= (args.min_steps1 if stage == 1 else args.min_steps2):
                    keys = ("align",) if stage == 1 else ("flow", "align")
                    done, gain = converged(hist, keys, args.conv_window, args.conv_thresh)
                    if gain:
                        print(f"[conv] stage {stage} step {step}: " + " ".join(f"{kk} {g:+.1%}" for kk, g in gain.items()),
                              flush=True)
                    streak = streak + 1 if done else 0
                    if streak >= args.conv_patience:
                        print(f"[conv] stage {stage} converged at step {step}", flush=True)
                        break
            if (time.time() - stage_start) / 3600 >= cap_h or elapsed >= args.budget_hours:
                print(f"stage {stage}: time cap reached at step {step} ({elapsed:.2f} h total)", flush=True)
                break
        save(f"stage{stage}_final")
        print(f"STAGE{stage}_DONE step {step} elapsed {(time.time() - t_start) / 3600:.2f}h", flush=True)
        if (time.time() - t_start) / 3600 >= args.budget_hours:
            break
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()

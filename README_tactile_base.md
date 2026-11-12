# tactile_base

pi0.5 + AnyTouch2 baseline on the [OpenViTac](https://github.com/FVL-Repo/OpenViTac) benchmark, built on top of
[openpi](https://github.com/Physical-Intelligence/openpi) (`upstream` remote).

This is a **re-implementation** of the OpenVTLA design described in the OpenViTac paper (arXiv 2610.10384): AnyTouch2
encodes the tactile history into compact tokens, which are projected to the VLM width and prepended to pi0.5's visual
token sequence (token-level concatenation). The authors' OpenVTLA code and checkpoints are not public, so the choices
below that the paper does not specify are ours.

| | Paper (OpenVTLA) | Here |
|---|---|---|
| Backbone | pi0.5, full fine-tune | pi0.5 (`pi05_base`), LoRA on Gemma 2B + 300M; SigLIP trained (openpi LoRA default) |
| Batch / steps | 64 / 20k | 64 (micro-batch 16 x 4 gradient accumulation) / 20k |
| Tactile encoder | AnyTouch2 (pretrained) | AnyTouch2 `checkpoint-4frames.pth`, **frozen**, features cached offline |
| Tactile history | not specified | 4 frames, stride 2 env steps, background = first tactile frame of the episode |
| Tokens | "compact" | per sensor: CLS + 2 (time) x 2x2 pooled patches = 9 -> 18 tokens for 2 sensors (`cls_pool2`) |
| Projector | "projected to VLM hidden dim" | LayerNorm -> Linear(768, 2048) -> GELU -> Linear(2048, 2048) |
| Actions | pi0.5 recipe | 8-d absolute joint target, trained as joint deltas (gripper absolute), horizon 50 |

## Code map

- `src/openpi/tactile/anytouch2.py` - frozen AnyTouch2 wrapper (shared by caching and serving) and token selection modes.
- `src/openpi/models/{model,pi0,pi0_config}.py` - `Observation.tactile`, tactile projector, prefix concatenation.
- `scripts/train.py`, `TrainConfig.grad_accum_steps` - gradient accumulation (scan over micro-batches).
- `src/openpi/policies/openvitac_policy.py` - OpenViTac input/output transforms (cached features in training, online
  AnyTouch2 encoding at inference).
- `src/openpi/training/config.py` - `pi05_vision_<task>` and `openvtla_<task>` configs (bottom of file).
- `examples/openvitac/convert_openvitac.py` - FVL-Lab/OpenViTac LeRobot v2.1 -> openpi dataset + AnyTouch2 cache.
- `third_party/AnyTouch2` - submodule (GeWu-Lab/AnyTouch2). Weights: gated HF repo `xxuan01/AnyTouch2-Model`.

## Usage

Tasks (simulation, GelSight Mini, 100 demos each): `insert_USB`, `grasp_chip`, `gear_assembly`,
`empty_can_selection` (configs `openvtla_<task>` and the vision-only `pi05_vision_<task>`; `hardness_classify` also
exists). AnyTouch2 weights need access to the gated HF repo `xxuan01/AnyTouch2-Model` (`huggingface-cli login`).

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync && git submodule update --init third_party/AnyTouch2
export HF_LEROBOT_HOME=/path/to/lerobot OPENPI_DATA_HOME=/path/to/openpi_cache
TASK=insert_USB
# 1. raw data (FVL-Lab/OpenViTac, revision used here: 5fb2757)
uv run huggingface-cli download FVL-Lab/OpenViTac --repo-type dataset --revision 5fb2757b88826917242adefd9196424ef8cf2d98 \
    --include "Simulation/GelSight_mini/$TASK/*" --local-dir /path/to/openvitac
# 2. openpi dataset + cached AnyTouch2 features (add --device cpu --dtype float32 if the GPU is busy)
uv run examples/openvitac/convert_openvitac.py --src /path/to/openvitac/Simulation/GelSight_mini/$TASK \
    --repo-id openvitac/${TASK}_gelsight
# 3. normalization stats, then train (paper protocol: 20k steps, batch 64, final checkpoint)
uv run scripts/compute_norm_stats.py --config-name openvtla_$TASK
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py openvtla_$TASK --exp-name=baseline --keep-period None
```

`--keep-period None` keeps only the latest checkpoint (~12 GB each); add `--no-wandb-enabled` without a wandb login.
Measured on one RTX 3090 (24 GB): ~20 s per optimizer step at batch 64 (~4.6 days per task).

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
- `src/openpi/training/config.py` - `pi05_vision_<task>`, `openvtla_<task>` (base) and `openvtla_<task>_margin` (ours)
  configs (bottom of file).
- `examples/openvitac/convert_openvitac.py` - FVL-Lab/OpenViTac LeRobot v2.1 -> openpi dataset + AnyTouch2 cache.
- `examples/openvitac/convert_dagger.py` - DAgger rollouts (hdf5) -> openpi dataset + AnyTouch2 cache + jam labels.
- `examples/openvitac/openvitac_client/` - OpenViTac-side additions: tactile history for the OpenPI client, DAgger
  collection, trajectory replay.
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

**Multi-task** (`openvtla_multitask`, `pi05_vision_multitask`): one policy on the four tasks above, every task drawn
equally often (`balance_datasets`); the four datasets are concatenated at load time (`DataConfig.repo_ids`, each sample
tagged with `dataset_index` to pick its AnyTouch2 cache), with normalization stats over the union. Convert the four
tasks first, then:

```bash
uv run scripts/compute_norm_stats.py --config-name openvtla_multitask
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py openvtla_multitask --exp-name=baseline --keep-period None
```

The paper trains single-task; the default budget (20k x 64) equals one single-task run, i.e. 1/4 of the samples per task.

**Ours** (`openvtla_<task>_margin`): the base OpenVTLA model (`openvtla_<task>`) with the same data, cached features,
architecture and hyper-parameters, plus three training-only losses (inference is unchanged):

| | Config field | Value |
|---|---|---|
| Contrastive tactile margin: flow loss with the real tactile must beat the loss with a wrong tactile (in-batch farthest) by a margin scaled by their feature distance | `tactile_margin_weight` / `tactile_margin_base` / `tactile_neg_mode` / `tactile_dref_pct` | 1.0 / 0.1 / `batch_far` / 50 |
| Vision feature mask on post-SigLIP image tokens | `vision_feature_mask_prob` | 0.1 |
| InfoNCE alignment of pooled tactile tokens to the clean wrist-camera (eef view) tokens | `tactile_align_weight` / `tactile_align_temp` | 0.1 / 0.1 |

The margin needs one extra LLM forward per step (~35 s per optimizer step on an RTX 3090, ~8 days per task). With all
three weights at 0 the loss reduces exactly to the base flow loss, so `openvtla_<task>` is the base for comparison.

```bash
uv run scripts/compute_norm_stats.py --config-name openvtla_${TASK}_margin   # same stats as openvtla_$TASK
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py openvtla_${TASK}_margin --exp-name=ours --keep-period None
```

The code also supports gating these losses by a per-frame contact score (`tactile/contact_checkpoint-4frames.npy` next
to the AnyTouch2 cache, picked up automatically if present). It is **not used**: do not create that file. (A gate built
from tactile-image change crossed its threshold at the 2nd frame of every insert_USB episode, i.e. it was all ones and
equivalent to no file.)

**DAgger** (`openvtla_insert_USB_dagger`, `openvtla_insert_USB_dagger_event`): fine-tunes the base model on scripted-expert
recoveries from the jammed insertion states the base policy itself reaches (collected in the simulator with
`examples/openvitac/openvitac_client/policy/openpi_dagger`, converted by `examples/openvitac/convert_dagger.py`), optionally
with a tactile event head that classifies "jammed" from the pooled tactile tokens. See `docs/dagger_insert_USB.md`
(background, data on HF, B200 commands).

`--keep-period None` keeps only the latest checkpoint (~12 GB each); add `--no-wandb-enabled` without a wandb login.
Measured on one RTX 3090 (24 GB): ~20 s per optimizer step at batch 64 (~4.6 days per task).

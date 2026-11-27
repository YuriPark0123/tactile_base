# Vision-only baseline: does the tactile modality help?

To check whether adding tactile input matters, we compare the base OpenVTLA model `openvtla_<task>` (pi0.5 + AnyTouch2)
with `pi05_vision_<task>`. Both models are trained on the same data with the same settings. The only difference is
that the vision-only model gets no tactile tokens.

| | `openvtla_<task>` (existing, HF `yrpark/openvtla-<task>`) | `pi05_vision_<task>` (new) |
|---|---|---|
| Inputs | head + wrist camera, state, prompt, **2 x 9 AnyTouch2 tokens (cls_pool2)** | head + wrist camera, state, prompt |
| Data | `openvitac/<task>_gelsight` (100 scripted demos) | same |
| Model | pi0.5, LoRA (gemma_2b_lora + gemma_300m_lora), horizon 50, joint delta actions | same |
| Training | 20k steps, batch 64, cosine LR 5e-5 → 5e-6 (warmup 1k), seed 42, from `pi05_base` | same |
| Norm stats | `assets/openvtla_<task>/...` | `assets/pi05_vision_<task>/...` (identical to OpenVTLA's, checked for all 4 tasks) |

Note: the OpenVTLA bases were trained on an RTX 3090 with batch 16 x 4 grad accum. On the B200 we use batch 64 x 1.
The effective batch is the same.

## B200 training (per task)

Tasks: `insert_USB`, `gear_assembly`, `empty_can_selection`, `grasp_chip`. Only the demo dataset is needed (no DAgger
data, no AnyTouch2 cache). The `pi05_base` weights are downloaded automatically from `gs://openpi-assets` into
`$OPENPI_DATA_HOME`.

```bash
cd tactile_base && git pull
export HF_LEROBOT_HOME=/path/to/lerobot OPENPI_DATA_HOME=/path/to/openpi_cache
TASK=grasp_chip   # insert_USB | gear_assembly | empty_can_selection | grasp_chip
uv run huggingface-cli download yrpark/openvitac-openvtla-data --repo-type dataset \
    --include "lerobot/openvitac/${TASK}_gelsight/*" --local-dir /tmp/ovt && rsync -a /tmp/ovt/lerobot/ $HF_LEROBOT_HOME/
# norm stats -> ./assets/pi05_vision_${TASK}/openvitac/${TASK}_gelsight/norm_stats.json (no compute_norm_stats needed)
uv run huggingface-cli download yrpark/openvitac-openvtla-data --repo-type dataset \
    --include "assets/pi05_vision_${TASK}/*" --local-dir .
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py pi05_vision_${TASK} --exp-name=baseline \
    --grad-accum-steps 1 --num-workers 8 --no-wandb-enabled
```

Upload (the same layout as `yrpark/openvtla-<task>`: `19999/params`, `19999/assets`, plus README and `train_log.txt`):

```bash
uv run huggingface-cli upload yrpark/pi05-vision-${TASK} checkpoints/pi05_vision_${TASK}/baseline/19999 19999 \
    --exclude "train_state/*"
```

Suggested order: `grasp_chip` and `insert_USB` first, because both already have a 100-episode OpenVTLA result.

## 3090 evaluation

The client configs `policy/openpi/abs_joint/deploy_pi05_vision_<task>.yml` are copies of
`deploy_openvtla_<task>.yml` (same prompt, temporal ensemble and chunk). The only change is `send_tactile: false`, so
inference also differs only in the tactile input. The older `deploy_vision.yml` / `deploy_grasp_chip_vision.yml` use
different settings (open-loop 20 steps, a different prompt), so do not use them for this comparison.

- Seeds 10000-10099 (100 episodes) for both models, with the same task configs as the OpenVTLA base evals (gear and
  insert_USB: `gelsight.yml`; empty_can: `gelsight_slowreset.yml`; grasp_chip: `grasp_chip.yml` + `run_force_seeds.py`).
- 20 episodes are too few to separate the two models (about ±2 episodes of sampling noise near 60% success).

| task | OpenVTLA base | pi0.5 vision |
|---|---|---|
| insert_USB | 24/100 | |
| gear_assembly | 17/20 (100-ep run stopped at 7/7) | |
| empty_can_selection | 12/20 | |
| grasp_chip | 92/100 | |

For a fair comparison, gear_assembly and empty_can_selection also need 100-episode OpenVTLA runs. The local
checkpoints were deleted (2026-11-27), so download `yrpark/openvtla-<task>` again before running them.

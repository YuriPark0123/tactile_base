# WorldTactile

Tactile world queries for a vision-language-action policy. WorldTactile adds touch to pi0.5 (openpi, PyTorch) in two
ways: the current tactile signal of each finger enters the VLM as a token, and learnable **tactile world queries**
are trained to predict the tactile signal of the near future. The queries attend bidirectionally to the action tokens
of the flow-matching action expert, so the actions being generated and the touch they will cause are modeled together.
Evaluated on the simulated visuo-tactile benchmark [UniVTAC](https://github.com/univtac/UniVTAC) (GelSight Mini, Franka).

This folder is self-contained (PyTorch); the OpenViTac tactile baseline at the repository root
(`README_tactile_base.md`, JAX) is separate. All commands below run from this folder.

## Method

**Tactile features.** A frozen [AnyTouch2](https://github.com/GeWu-Lab/AnyTouch2) encoder maps each finger's 4-frame
clip (t-12, t-8, t-4, t at 60 Hz) to its projected CLS (512-d), standardized per dimension over the training data.

**Tokens.** VLM stream: `[head image 256 | wrist image 256 | (masked 3rd image) | prompt 200 | tactile L, R | queries
L x4, R x4]`; action expert: `[50 actions]`. Tactile tokens are `Linear(CLS) + finger embedding`.

**Attention** (row attends to column):

| query \ key | images + prompt | tactile | own finger's queries | other finger's queries | actions |
|---|---|---|---|---|---|
| images + prompt | yes | – | – | – | – |
| tactile | yes | yes | – | – | – |
| queries | yes | yes | yes | – | yes |
| actions | yes | yes | yes | yes | yes |

Images, prompt and tactile never see queries or actions, so at inference their keys and values are computed once per
chunk and each of the 10 flow steps reruns only the queries and the actions.

**Losses.**
- `flow`: pi0.5 flow matching on the 9-d robot action chunk (H = 50).
- `align`: a shared MLP on each finger's 4 query states predicts the AnyTouch2 CLS of the future clip
  (t+12, t+16, t+20, t+24); MSE against the stop-gradient target.
- `adhere` (weight 0.1): `max(0, m - (flow(corrupted tactile) - flow))`, the contextual adherence loss of RA-VLA
  (Eq. 6) with tactile as the context; the corrupted tactile is the resting sensor or a contact frame of another task.

**Training.** Stage 1 trains only the new modules (queries, tactile input, prediction head) on `align`. Stage 2 adds a
LoRA on all PaliGemma layers and fully trains the action expert and the action/time projections on
`flow + align + 0.1 * adhere`. SigLIP and the base VLM weights stay frozen. Training samples are the contact-rich part
of each demonstration (first contact - 0.5 s to last contact + 0.5 s), tasks sampled equally often.

**Actions.** `action[t] = joint[t+1]` (7 arm + 2 finger joints). Arm joints are chunk-wise deltas
`joint[t+k] - joint[t]`; fingers are absolute. Quantile normalization to [-1, 1] (openpi convention), with per-task,
per-chunk-index statistics for the deltas.

**Inference.** The server returns a 50-step chunk; the adapter executes the first 25 and re-plans. Optional real-time
chunking (RTC) guidance toward the previous chunk and overlap blending (`--rtc`, `--rtc_blend`).

## Repository layout

```
worldtactile/
  paths.py            external files and directories (environment variables, see below)
  univtac.py          reading UniVTAC isaac51 HDF5 episodes
  anytouch2.py        frozen AnyTouch2 encoder
  normalization.py    state / action normalization
  dataset.py          contact-rich training samples
  pi05/model.py       WorldTactilePi05: tokens, attention mask, joint layer loop, losses, cached inference, RTC
  pi05/collate.py     openpi image / prompt conventions
  serving/            policy server: wire protocol, policy, closed-loop tactile prediction log
scripts/
  convert_openpi_checkpoint.py   pi05_base JAX -> PyTorch
  preprocess_univtac.py          contact ranges + AnyTouch2 CLS for every frame
  train.py                       two-stage training
  serve.py                       policy server for a simulator adapter
  check_model.py                 numerical checks of the model (vs openpi, cache, masks, gradients)
resources/gsmini_sim_background_rgb.png   UniVTAC's simulated GelSight Mini image without contact
```

## Setup

Tested with Python 3.11, PyTorch 2.7.1 (CUDA 12.6), one RTX 3090 (24 GB).

1. **openpi** (PyTorch model code): the openpi of this repository (root `src/openpi`, based on upstream commit
   `215abfb`; WorldTactile uses only its unmodified PyTorch model). Install it and apply its transformers patch, as in
   openpi's README:
   ```bash
   cd <repo root> && pip install -e .
   cp -r src/openpi/models_pytorch/transformers_replace/* $(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")/
   ```
2. Other packages: `pip install -e worldtactile` (transformers 4.53.2, peft 0.21.2, sentencepiece, h5py,
   opencv-python-headless, pillow, scipy, safetensors).
3. **pi0.5 base**: download openpi's `pi05_base` checkpoint and the PaliGemma tokenizer
   (`paligemma_tokenizer.model`), then convert (run from this folder):
   ```bash
   python scripts/convert_openpi_checkpoint.py --openpi .. --src <pi05_base dir> --dst <out dir>
   ```
4. **AnyTouch2**: clone the repository (tested at `82c5677`) and place `checkpoints/checkpoint-4frames.pth` and
   `CLIP-B-16/config.json` as in its instructions.
5. **UniVTAC data**: the `isaac51/<task>/hdf5/*.hdf5` episodes of the Hugging Face dataset `byml/UniVTAC`, and the
   UniVTAC repository for `instructions/<task>.json`.

Point the code at these locations with environment variables (defaults are this project's server layout):

| Variable | Content |
|---|---|
| `WT_UNIVTAC_RAW` | UniVTAC `isaac51` folder (`<task>/hdf5/*.hdf5`) |
| `WT_UNIVTAC_INSTRUCTIONS` | UniVTAC `instructions/` folder |
| `WT_PROCESSED` | output of `preprocess_univtac.py` |
| `WT_ANYTOUCH2_ROOT` | AnyTouch2 repository |
| `WT_PI05_BASE` | converted pi0.5 base (PyTorch) |
| `WT_PALIGEMMA_TOKENIZER` | `paligemma_tokenizer.model` |
| `WT_EXP_ROOT` | training runs |
| `WT_GEL_BACKGROUND` | no-contact gel image (default: `resources/`) |

## Usage

```bash
# 1. preprocess (per task)
for t in insert_hole insert_HDMI put_bottle_in_shelf; do python scripts/preprocess_univtac.py --task $t; done

# 2. train: stage 1 until convergence, then stage 2 (36 h budget)
python scripts/train.py --run_name pi05_wt_delta --lora_rank 256
#    our main checkpoint continued stage 2 from that run's stage-2 result for another 25 h
python scripts/train.py --run_name pi05_wt_delta_s2b --lora_rank 256 --budget_hours 25 --conv_window 2000 \
    --min_steps2 4000 --conv_patience 2 --resume $WT_EXP_ROOT/pi05_wt_delta/trainable_stage2_final.pt \
    --start_stage 2 --start_step 10000

# 3. serve to a simulator adapter (prints "READY on 127.0.0.1:6123")
python scripts/serve.py --run $WT_EXP_ROOT/pi05_wt_delta_s2b --step stage2_final --port 6123 \
    [--query_log <dir>/query_log.jsonl] [--rtc 0 --rtc_blend 0]

# model checks (needs preprocessed insert_HDMI and insert_hole)
python scripts/check_model.py
```

A run directory holds `args.json`, `norm_stats.npz`, `cls_stats.npz` and `trainable_<tag>.pt` (the trained
parameters only: new modules, LoRA, action expert, action/time projections; the rest comes from pi0.5 base).

**Simulator side.** `serve.py` speaks the protocol in `worldtactile/serving/protocol.py` (localhost
`multiprocessing.connection`, npz messages). The UniVTAC adapter that sends observations and executes the returned
joint targets is not part of this repository.

**Tactile prediction log.** With `--query_log`, each request logs the predicted future CLS, and when the adapter later
sends the realized clip, the prediction error against it and against the copy baseline (current tactile); raw vectors
and per-request attention mass by modality are saved next to the log.

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (pi0.5), [AnyTouch2](https://github.com/GeWu-Lab/AnyTouch2)
and [UniVTAC](https://github.com/univtac/UniVTAC). The adherence loss follows RA-VLA (contextual adherence, Eq. 6); the
RTC guidance follows Black et al., "Real-Time Execution of Action Chunking Flow Policies" (2025).

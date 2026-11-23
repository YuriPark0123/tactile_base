# DAgger: 정책이 실패하는 상태에서의 전문가 회복 데이터로 fine-tune

공통 파이프라인 (task별 차이는 OpenViTac 쪽 플러그인 하나):

1. **base 평가** (3090, 시뮬레이터, 20 episode, seed 10000부터): 실패 영상과 privileged 상태로 실패 지점 파악.
2. **DAgger 수집** (`examples/openvitac/openvitac_client/policy/openpi_dagger`, seed 20000부터 — 평가 seed와 겹치지
   않게): base 정책을 그대로 실행하다 task 플러그인(`tasks/<task>.py`)이 실패 상태를 감지하면 그 task의 스크립트
   primitive로 전문가가 개입. 모든 프레임 기록 (`dagger/<seed>_<result>.hdf5`: 이미지 4종, state, action, source,
   `event_rule`, `goal`, 플러그인 진단값).
3. **변환** (`examples/openvitac/convert_dagger.py --event-name <name>`): `goal`에 도달한 전문가 회복 구간 + 개입 없이
   성공한 episode만 학습 데이터로. tactile event 라벨 1 = 전문가에게 넘겨진 실패 상태, −1 = 애매(loss 제외), 0 = 나머지.
4. **학습** (B200): `openvtla_<task>_dagger` (BC) / `openvtla_<task>_dagger_event` (+ tactile event head) —
   base 체크포인트에서 5k step fine-tune, base norm stats, 시연:DAgger = 50:50.
5. **평가**: 3090에서 seed 10000부터 20 episode, base와 비교.

플러그인 인터페이스 (`Expert`): `diag(task)` 진단값, `trigger(task, d)` 개입 사유, `recover(task, reason)` 스크립트
회복, `event_rule(d)` 라벨 규칙, `goal_reached(d)` 회복 성공, `reset()`, `after_recover(task)`.

## Task별 상태

| task | base (20 ep) | 실패 원인 | 플러그인 / 이벤트 | 데이터 (HF `yrpark/openvitac-openvtla-data`) | 문서 |
|---|---|---|---|---|---|
| insert_USB | 3/20 (100 ep: 24%) | 슬롯 입구에서 어긋난 채 걸림 | `insert_USB` / `jam` | `lerobot/openvitac/insert_USB_dagger_gelsight`, `raw/insert_USB_dagger_r1` | `docs/dagger_insert_USB.md` |
| gear_assembly | 17/20 | 빨간 기어가 축 위 ~4.6 mm에서 톱니에 얹혀 막힌 채 회전 시작 | `gear_assembly` / `blocked` | `lerobot/openvitac/gear_assembly_dagger_gelsight`, `raw/gear_assembly_dagger_r1` | 아래 |

## B200 학습 (공통)

```bash
cd tactile_base && git pull
export HF_LEROBOT_HOME=/path/to/lerobot OPENPI_DATA_HOME=/path/to/openpi_cache
TASK=gear_assembly
uv run huggingface-cli download yrpark/openvitac-openvtla-data --repo-type dataset \
    --include "lerobot/openvitac/${TASK}_gelsight/*" "lerobot/openvitac/${TASK}_dagger_gelsight/*" \
    --local-dir /tmp/ovt && rsync -a /tmp/ovt/lerobot/ $HF_LEROBOT_HOME/
uv run huggingface-cli download yrpark/openvtla-$TASK --include "19999/params/*" "19999/assets/*" \
    "19999/_CHECKPOINT_METADATA" --local-dir checkpoints/openvtla_$TASK/baseline
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py openvtla_${TASK}_dagger_event --exp-name=r1 \
    --grad-accum-steps 1 --num-workers 8 --no-wandb-enabled
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py openvtla_${TASK}_dagger --exp-name=r1 \
    --grad-accum-steps 1 --num-workers 8 --no-wandb-enabled
```

repo 루트에서 실행 (`checkpoints/openvtla_<task>/baseline/19999`는 상대 경로). norm stats 계산 불필요.

## 3090 시뮬레이터 메모

- `--eval_step_timeout_seconds 0` (접촉이 많은 UIPC step이 기본 5초 제한을 넘음).
- gear_assembly / empty_can_selection은 reset이 120초 기본 제한을 넘음 → `task_config/gelsight_slowreset.yml`
  (`reset_time_limit: 600`, `examples/openvitac/openvitac_client/task_config/`).
- 강제 종료한 Isaac Sim이 GPU를 놓기 전에 다음 실행을 시작하면 매우 느려짐 (step당 ~30초): `nvidia-smi` 확인 후 시작.

## gear_assembly

- base `yrpark/openvtla-gear_assembly`: 17/20 (실패 seed 10002, 10016, 10018). 실패 episode: 빨간 기어가 조립 높이보다
  ~4.6 mm 위에서 멈춤 (잡을 때의 yaw가 톱니 위상과 ~2° 어긋나 파란 기어 톱니 위에 얹힘). 정책은 그대로 45° 회전을
  시작하고, 반쯤 맞물린 채 돌다 빨간 기어가 3~5 mm 밀려 파란 기어가 30° 미만 회전 (성공 기준 30°).
- 개입 (`tasks/gear_assembly.py`): 축 6 mm 이내에서 2.5~10 mm 높이에 10 policy step 동안 0.3 mm 이상 못 내려가면
  (`blocked_above_shaft`), 또는 축 높이에서 중심이 2.5 mm 이상 어긋나면 (`seated_off_centre`).
- 회복: 파란 기어가 이미 돌아간 만큼 빨간 기어 목표 위상 보정 (1:1 반대 방향) → 스크립트 재삽입
  (`_insert_red_gear_onto_base`) → 여전히 1.5 mm 이상 떠 있으면 놓고 다시 집어 재삽입 → 시연과 같은 45° 회전
  (기어 구동 포함). episode당 최대 1회.
- 라벨 `event_blocked`: 축 근처에서 막힌 높이(2.5~10 mm) 또는 중심 어긋남.

수집 r1 (3090, seed 20000~20031, 50개 계획 중 32개에서 중단 — 한 episode 평균 ~9분):

| | episode |
|---|---|
| 개입 없이 base 성공 | 4 |
| 개입 → 회복 목표 도달 (기어 안착 + 한 쌍 회전) | 16 (그중 task 성공 판정 10) |
| 개입 → 회복 실패 | 12 (대부분 재파지 후 기어가 크게 틀어짐: 높이 15 mm, 위상 10°) |

학습 데이터: 8,593 프레임 = 회복 구간 16개 + base 성공 4개. 라벨 1: 169 (구간마다 10~12), −1: 3,829, 나머지 0.
3090 확인: 데이터 로딩 (시연 27,324 + DAgger 8,593, 라벨 0/1/−1 정상). 학습 step은 insert_USB config와 같은 경로
(GPU 테스트는 insert_USB에서 통과).

한계와 다음 라운드:
- 개입이 32개 중 28개로 많음: 정책이 축 위에서 잠깐 멈추는 정상 동작도 "막힘"으로 감지 (base 단독 성공률 85%).
  `stuck_window`를 늘리면 개입이 줄어 정말 막힌 상태만 남지만 수집 시간이 늘어남.
- 재파지 경로의 회복 성공률이 낮음 → 회복 실패 12개는 학습 데이터에서 제외됨. 재파지 대신 작은 yaw 흔들기(wiggle)로
  위상을 찾는 회복이 tactile 활용 관점에서도 더 자연스러움 (미구현).

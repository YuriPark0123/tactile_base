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
| gear_assembly | 17/20 | 빨간 기어가 축 위 ~4.6 mm에서 톱니에 얹혀 막힌 채 회전 시작 | `gear_assembly` / `blocked` | `lerobot/openvitac/gear_assembly_dagger_gelsight` (r1+r2), `raw/gear_assembly_dagger_r{1,2}` | 아래 |
| grasp_chip | 19/20 | 접시에 닿는 순간 충돌 (`impact`); DAgger 수집 중 추가로: 운반 중 그리퍼가 벌어져 칩이 빠짐 | `grasp_chip` / `impact` | `lerobot/openvitac/grasp_chip_dagger_gelsight`, `raw/grasp_chip_dagger_r1` | 아래 |
| empty_can_selection | 12/20 | 캔 무게 오판 (빈 캔을 무겁다고 내려놓거나 무거운 캔을 바구니로) | `empty_can_selection` / `empty` (dense) | `lerobot/openvitac/empty_can_selection_dagger_gelsight` (r1+r2), `raw/empty_can_selection_dagger_r{1,2}` | 아래 |

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
- 전문가 개입 중에는 `task.in_pre_move = True`로 둠: 평가 모드의 `BaseTask.move()`는 pre-move 단계가 아니면 그리퍼
  목표를 비율이 아닌 qpos로 해석해서 `close_gripper(0.7)` 같은 스크립트 동작이 그리퍼를 닫지 않음 (수집기에서 처리).
  gear_assembly r1은 이 수정 전에 수집되어 재파지 회복이 대부분 실패함 (실패 회복은 학습 데이터에서 제외됨).
- 같은 seed, 같은 코드라도 실행(launch)마다 UIPC solver가 빠른 상태와 느린 상태로 갈림 (느리면 step당 수 초, gear reset이
  600초 제한 초과). 실행 단위 현상이라 프로세스를 새로 띄우면 대개 정상 속도: gear 수집은 reset 시간 초과 예외가 나면
  다음 seed부터 다시 띄움, grasp_chip은 `scripts/run_force_seeds.py` (120초 뒤에도 최근 10 step이 step당 2초보다 느리면 같은 seed 재실행; 정상 실행도 처음 ~15 step은 느릴 수 있음).
- 강제 종료한 Isaac Sim이 GPU를 놓기 전에 다음 실행을 시작하면 매우 느려짐 (step당 ~30초): `nvidia-smi` 확인 후 시작.

## gear_assembly

- base `yrpark/openvtla-gear_assembly`: 17/20 (실패 seed 10002, 10016, 10018). 실패 episode: 빨간 기어가 조립 높이보다
  ~4.6 mm 위에서 멈춤 (잡을 때의 yaw가 톱니 위상과 ~2° 어긋나 파란 기어 톱니 위에 얹힘). 정책은 그대로 45° 회전을
  시작하고, 반쯤 맞물린 채 돌다 빨간 기어가 3~5 mm 밀려 파란 기어가 30° 미만 회전 (성공 기준 30°).
- 개입 (`tasks/gear_assembly.py`): 축 6 mm 이내에서 2.5~10 mm 높이에 10 policy step 동안 0.3 mm 이상 못 내려가면
  (`blocked_above_shaft`), 또는 축 높이에서 중심이 2.5 mm 이상 어긋나면 (`seated_off_centre`).
- 회복 (r2 최종): 파란 기어가 이미 돌아간 만큼 빨간 기어 목표 위상 보정 (1:1 반대 방향) → 스크립트 재삽입
  (`_insert_red_gear_onto_base`) → 30 step 대기 후 아직 2 mm 이상 떠 있으면 **wiggle**: 쥔 채로 +2/−2/+4/−4/+6/−6°
  (톱니 피치 12°의 절반까지) 돌리며 살짝 눌러 맞물리는 위상 탐색 → 그래도 떠 있으면 놓기 (놓으면 스스로 미끄러져
  안착하는 경우가 많음) → 제자리에서 다시 잡기 → 그래도 떠 있을 때만 재삽입 → 시연과 같은 45° 회전 (기어 구동 포함).
  episode당 최대 1회.
- 라벨 `event_blocked`: 축 근처에서 막힌 높이(2.5~10 mm) 또는 중심 어긋남.

수집 (3090):

| | r1 (seed 20000~20031) | r2 (seed 20032~20061) |
|---|---|---|
| 개입 없이 base 성공 | 4 | 1 |
| 개입 → 회복 목표 도달 (기어 안착 + 한 쌍 회전) | 16 | 23 (최종 플러그인 seed 20041~: 19/21) |
| 개입 → 회복 실패 | 12 | 6 (그중 4개는 아래 수정 전) |

r1 회복 실패 원인: 평가 모드 그리퍼 해석 문제 (재파지에서 그리퍼가 닫히지 않음, 수정됨). r2 중 고친 것:
(1) 삽입 직후 기어가 아직 천천히 내려가는 중에 판정해서 안착한 기어를 다시 들어 올림 (20032, 20035) → 30 step 대기
후 판정, 재삽입 전에 먼저 놓아 봄. (2) 재파지 기준 2.5 mm가 안착 기준 2 mm보다 커서 2.3 mm에 멈춘 채 회전 (20038)
→ 2 mm. (3) 삽입 중 파란 기어가 접촉으로 2~3° 더 돌아 계산한 위상이 틀림 (20040) → wiggle.
task 판정 (`check_success`)의 `red_gear_insert_success`는 스크립트 삽입 직후에 한 번 기록되므로, wiggle/재파지로 나중에
안착한 episode는 기어가 맞물려 돌았어도 실패로 남음 (r2의 회복 목표 도달 23개 중 task 성공 14). 학습 데이터 선별은
물리 상태로 계산한 `goal`을 씀.

학습 데이터 (r1+r2): 15,118 프레임 = 회복 구간 39개 (13,878 프레임) + base 성공 5개 (1,240 프레임). 라벨 1: 406,
0: 10,007, −1: 4,705. 회복 실패 18개는 제외.

한계: 개입이 많음 (62개 중 57개) — 정책이 축 위에서 잠깐 멈추는 정상 동작도 "막힘"으로 감지 (base 단독 성공률 85%).
그래서 학습 데이터 대부분이 전문가 회복 구간이고 base 성공 구간이 적음 (시연 데이터와 50:50으로 섞어 보완).

## empty_can_selection

- base `yrpark/openvtla-empty_can_selection`: 12/20. 실패 8개 모두 무게 오판: 빈 캔(환타)을 들어 보고 무겁다고 내려놓은
  뒤 무거운 펩시를 바구니에 (3), 무거운 환타를 바구니에 (5, 빈 캔 = 펩시). 두 캔은 외형이 같고 들었을 때의 하중만 다름.
- 개입 (`tasks/empty_can_selection.py`): 무거운 캔을 든 채 제자리에서 3 cm 이상 옮김 (`carrying_heavy`), 또는 빈 캔을
  2.5 cm 이상 들어 본 뒤 제자리 근처에서 놓기 시작함 — 손가락이 쥔 폭보다 0.8 mm 이상 벌어지거나 1.2 cm 아래까지 내림
  (`releasing_empty`, r2부터; r1의 `returning_empty`는 캔이 바닥에 떨어진 뒤에야 발동).
- 회복 (스크립트 primitive): `carrying_heavy` → 무거운 캔 제자리 → 빈 캔 잡기 → 들기 → 바구니; `releasing_empty` → 다시
  쥐기 (이미 빠졌으면 다시 잡기) → 들기 → 바구니 → 20 step 대기 (캔이 바구니에 자리 잡을 때까지 기록).
- 라벨 `event_empty` (dense, `--event-mode dense`): 캔을 들고 있는 프레임마다 1 = 빈 캔, 0 = 무거운 캔, 그 외 −1.
  tactile이 하중 차이를 직접 구분하도록 학습시키는 신호 (BCE 양성 가중치 0.15, 클래스 비율 보정). 이 라벨은 시연 데이터에는 없으므로
  시연 프레임은 −1 (`event_default`).

수집 (3090, episode당 ~2~3분):

| | r1 (seed 20000~20049) | r2 (seed 20050~20099) |
|---|---|---|
| 개입 없이 base 성공 | 31 | 27 |
| `carrying_heavy` 교정 → 성공 | 10/10 | 7/7 |
| 빈 캔 놓기 교정 → 성공 | 0/9 (`returning_empty`, 늦게 발동) | 13/13 (`releasing_empty`) |
| 개입 없이 실패 | 0 | 3 (빈 캔을 바구니 위로 옮겼지만 step 제한까지 놓지 않음 — 무게 판단과 무관) |

r1의 `returning_empty` 실패 원인: base 정책은 빈 캔을 ~5 cm 들어 본 뒤 그 높이에서 그리퍼를 열어 캔을 떨어뜨림. 규칙이
캔이 바닥에 닿은 뒤에 발동해서 전문가의 "다시 들기"가 빈 그리퍼를 들어 올림. r2는 손가락이 벌어지기 시작하는 순간에
개입해서 캔이 아직 손에 있음.

학습 데이터 (r1+r2, `--event-mode dense --event-name empty --goal-from-episode-success`): 20,984 프레임 = 교정 구간 30개
(6,760 프레임) + base 성공 58개 (14,224 프레임). 라벨 1 (빈 캔): 11,270, 0 (무거운 캔): 1,638, −1: 8,076.
`--goal-from-episode-success`: 놓은 캔이 마지막 전문가 프레임에서 아직 떨어지는 중이라 `goal`이 0인 경우가 있어서, episode가
성공했으면 마지막 전문가 구간을 회복 성공으로 인정 (r2 이후 플러그인은 놓은 뒤 20 step 기록하므로 불필요).

한계: 라벨 0 (무거운 캔을 든 프레임)이 1,638로 적음 — 무거운 캔은 들자마자 내려놓기 때문. `DAGGER_TASKS`의 BCE 양성 가중치를 0.15로 맞춤 (1,638 / 11,270).

## grasp_chip

force task라 seed마다 별도 프로세스로 실행: `scripts/run_force_seeds.py` (OpenViTac `scripts/`에 복사; 느린 solver 상태에
빠진 실행은 다시 띄움). 평가·수집 모두 이 러너로:

```bash
python scripts/run_force_seeds.py --out eval_result/openpi/grasp_chip/base20 --seeds 10000-10019 -- \
    grasp_chip task_config/grasp_chip.yml policy/openpi/abs_joint/deploy_openvtla_grasp_chip.yml --eval_step_timeout_seconds 0 --headless
python scripts/run_force_seeds.py --out eval_result/openpi_dagger/grasp_chip/r1 --seeds 20002-20031 -- \
    grasp_chip task_config/grasp_chip.yml policy/openpi_dagger/deploy_grasp_chip.yml --eval_step_timeout_seconds 0 --headless
```

- base `yrpark/openvtla-grasp_chip`: 19/20 (seed 10017 `impact`: 쥔 칩이 접시에 너무 빨리 닿음). 성공 episode도 여유가 작음
  (최대 변형률이 파손 기준의 74%까지, 충돌 속도 0.06/0.09 m/s까지) → 100 episode로 다시 측정 중.
- replay: 평가의 `trajectories/<seed>_<result>.npz` (실행한 abs qpos 액션)를 정책 서버 없이 다시 실행
  (`policy/replay_traj/deploy_grasp_chip.yml`, 같은 러너로). 시험 3/3 (10017 impact 실패, 10011·10019 성공): 결과·step 수
  동일, 관절 상태 차이 0, 칩 최대 변형률·충돌 속도 1e-12 이내로 같음. 각 replay는 `replay_check/<seed>.json`에 원래 결과와
  상태 차이를 기록하고 영상을 새로 저장:

  ```bash
  python scripts/run_force_seeds.py --out eval_result/replay_traj/grasp_chip/run1 --seeds 10017-10017 -- \
      grasp_chip task_config/grasp_chip.yml policy/replay_traj/deploy_grasp_chip.yml --eval_step_timeout_seconds 0 --headless
  ```
- 개입 (`tasks/grasp_chip.py`, policy 단계에서만):
  - `slipping`: 들어 올린 칩을 접시 밖에서 운반하는 중 손가락이 가장 좁게 쥔 폭보다 0.15 mm 이상 벌어짐 (채점기의 "벌어짐"
    기준과 같음). 수집 중 발견한 실패 유형 (seed 20006: 0.2 mm 벌어진 뒤 15 프레임 만에 9 cm 높이에서 빠져 바닥 충돌).
    성공 episode의 운반 중 최대 벌어짐은 0.06 mm.
  - `fast_descent`: 쥔 칩이 접시 위 2.5 cm 이내에서, 스크립트의 감속(0.9 m/s²)으로 멈춰도 접촉 순간 0.05 m/s 이상 남을 속도로
    내려감 (충돌 파손 기준 0.09 m/s). 스크립트 자체도 0.16 m/s로 내려와 접시 1.5 mm 위에서 멈추므로 단순 속도 기준은 쓰지 않음.
  - `release_high`: 접시 위 4 mm 이상에서 손가락이 벌어지기 시작.
- 회복 (스크립트 primitive): 다시 쥐기 (정책이 쥐었던 가장 좁은 폭으로 닫은 뒤 스크립트의 단계적 tactile 닫기) → (`slipping`)
  들기·접시 위로 운반 → 접시 위 접근 높이까지 천천히 → 촉각 마커 기준선 → 지지 접촉까지 6 mm/s로 내려놓기 → 접촉 유지 →
  놓기 → 빼기. 정책이 잡은 칩에는 스크립트의 손 안 자세가 없으므로 개입 시점 상태로 채움.
- 라벨 `event_impact` (onset): 전문가에게 넘겨진 실패 상태 프레임.

수집 r1 (3090, seed 20002~20031, 30 episode, episode당 ~5분 + 느린 실행 재시도):

| | episode |
|---|---|
| 개입 없이 base 성공 | 22 |
| `slipping` 교정 → 성공 | 4 (실패 1: 다시 쥐기 강화 전, 내려가다 다시 빠짐) |
| `fast_descent` 교정 → 성공 | 1 |
| 개입 없이 실패 | 2 (seed 20006: `slipping` 트리거 추가 전 / seed 20025: 칩을 끝내 못 집고 시간 초과 — 교정 대상 아님) |

학습 데이터: 11,501 프레임 = 교정 구간 5개 (2,384 프레임) + base 성공 22개 (9,117 프레임). 라벨 1은 4 프레임뿐
(개입이 드물고 실패 상태가 1~2 프레임) → grasp_chip은 event head가 사실상 학습되지 않으므로 `openvtla_grasp_chip_dagger`
(BC)를 기본으로. 데이터 대부분이 정책 자신의 성공 rollout이라 교정 효과보다는 on-policy 성공 데이터 추가에 가까움.

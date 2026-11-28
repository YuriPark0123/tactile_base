# 평가 결과 종합 (baseline / DAgger / vision-only)

모든 평가는 3090 시뮬레이터에서 seed 10000부터 연속으로 돌렸습니다. 100회는 seed 10000-10099, 20회는 seed 10000-10019입니다.
**20회 결과는 오차가 큽니다 (성공률 60% 근처에서 ±2회 정도).** 100회로 다시 평가하면 이 문서를 갱신합니다.

| 모델 | 설명 | HF |
|---|---|---|
| baseline (OpenVTLA) | pi0.5 + AnyTouch2 촉각 토큰, demo 100개로 20k step 학습 | `yrpark/openvtla-<task>` |
| DAgger | baseline에서 시작해 demo:DAgger 데이터를 50:50으로 섞어 5k step 추가 학습 | `yrpark/openvtla-<task>-dagger-<r>` |
| DAgger event | DAgger에 촉각 이벤트 분류 head를 더함 (BCE loss, 가중치 0.1, 학습할 때만 사용) | `yrpark/openvtla-<task>-dagger_event-<r>` |
| vision-only | baseline과 같은 데이터와 설정에서 촉각 토큰만 뺌 (`pi05_vision_<task>`) | `yrpark/pi05-vision-<task>` |

## 1. 성공률 요약 (task별로 가장 좋은 결과)

| task | baseline | 가장 좋은 DAgger | vision-only |
|---|---|---|---|
| insert_USB | 24/100 (24%) | dagger r1 **14/20 (20회)** | 학습 전 |
| gear_assembly | 84/100 (84%) | dagger_event_still r3 16/20, dagger_still r3 13/20 **(20회)** | 학습 전 |
| empty_can_selection | 56/100 (56%) | dagger r2 **16/20 (20회)** | 학습 전 |
| grasp_chip | 92/100 (92%) | dagger r2 19/20, dagger_event r2 19/20 **(20회)** | 84/99 (85%) |

## 2. baseline vs DAgger (같은 seed 10000-10019, 20회)

baseline 열은 100회 평가 중 처음 20개 seed의 결과입니다.

| task | baseline | dagger r1 | dagger_event r1 | dagger r2 | dagger_event r2 | dagger_still r3 | dagger_event_still r3 |
|---|---|---|---|---|---|---|---|
| insert_USB | 3/20 | **14/20** | 10/20 | – | – | – | – |
| gear_assembly | **17/20** | 8/20 | 5/20 | 3/20 | 6/20 | 13/20 | 16/20 |
| empty_can_selection | 12/20 | 13/20 | 14/20 | **16/20** | 12/20 | – | – |
| grasp_chip | 19/20 | – | – | 19/20 | 19/20 | – | – |

- insert_USB: DAgger로 크게 좋아짐 (3/20 → 14/20).
- gear_assembly:
  - r1/r2는 크게 떨어짐. 회복 스크립트가 기다리는 동안의 정지 프레임을 학습해서 기어를 끼운 뒤 돌리지 않고 멈췄습니다.
  - r3는 정지 프레임을 잘라낸 데이터로 학습했고, 대부분 회복했지만 baseline보다는 아직 낮습니다.
  - r3 실패는 모두 기어 삽입은 성공한 뒤 거의 회전하지 않은 경우입니다 (0-8도).
- empty_can_selection: r2 dagger가 가장 좋음 (12/20 → 16/20).
- grasp_chip: baseline이 이미 19/20이라 차이를 확인할 수 없음.
- event와 일반 DAgger의 차이 (event − 일반, 성공 횟수): USB r1 −4, gear r1 −3, gear r2 +3, gear r3 +3, can r1 +1, can r2 −4, chip r2 0. 일관되지 않고 20회 기준으로는 오차 범위 안입니다.

## 3. baseline vs vision-only (촉각의 효과, 100회)

| task | baseline (촉각 포함) | vision-only | 비고 |
|---|---|---|---|
| grasp_chip | 92/100 | 84/99 | McNemar p ≈ 0.04 |
| insert_USB | 24/100 | – | 학습 전 |
| gear_assembly | 84/100 | – | 학습 전 |
| empty_can_selection | 56/100 | – | 학습 전 |

- grasp_chip: 같은 seed로 비교하면 vision만 실패한 경우가 8회, baseline만 실패한 경우가 1회, 둘 다 실패한 경우가 7회입니다.
  - 실패 유형 (vision / baseline): 시간 초과 6/2, 충돌 5/4, 과도하게 쥠 4/2.
  - seed 10030은 시뮬레이터가 8번 재시도 모두 느린 상태에서 벗어나지 못해 제외했습니다.

## 4. baseline 실패 분석 (100회)

- gear_assembly (실패 16회): 모두 기어 삽입은 성공했습니다. 하지만 파란 기어가 16-29도만 돌았습니다 (30도 이상 필요). 기어 중심이 약 4 mm 밀려 맞물림이 미끄러졌습니다 (성공한 경우는 1 mm 미만).
- empty_can_selection (실패 44회):
  - 39회는 캔을 잘못 골랐습니다. 25회는 빈 첫 번째 캔을 내려놓고 두 번째 캔을 가져갔고, 14회는 무거운 첫 번째 캔을 가져갔습니다.
  - 5회는 올바른 캔을 골랐지만 바구니에 넣지 못했습니다.
  - 올바른 캔을 고른 비율은 61%이고, 고른 경우의 성공률은 92%입니다. 무게를 촉각으로 구분하는 단계가 병목입니다.
- grasp_chip (실패 8회): 충돌 4, 과도하게 쥠 2, 시간 초과 2.
- insert_USB: 24/100.

## 진행 중 / 예정

- 일반 DAgger 중 task별로 가장 좋은 모델을 100회로 늘리는 중입니다. 이미 돌린 seed 10000-10019에 10020-10099를 더합니다.
  - 순서: insert_USB dagger r1 → empty_can dagger r2 → grasp_chip dagger r2 → gear dagger_still r3.
- vision-only insert_USB / gear_assembly / empty_can_selection: B200에서 학습이 끝나면 각각 100회 평가합니다.

자세한 내용: [dagger.md](dagger.md), [vision_baseline.md](vision_baseline.md).

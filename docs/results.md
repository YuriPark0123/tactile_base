# 평가 결과 종합

3090 시뮬레이터, seed 10000부터 연속 평가 (100회 = 10000-10099, 20회 = 10000-10019). **(20회)** 표시는 아직 20회만 평가한
결과로, 100회 평가가 끝나면 갱신합니다. baseline 괄호 안의 수치는 같은 20개 seed (10000-10019)에서의 결과입니다.

| task | baseline (OpenVTLA) | DAgger | DAgger + event head | vision-only |
|---|---|---|---|---|
| insert_USB | 24/100 (20 seed: 3/20) | 14/20 (20회) | 10/20 (20회) | 학습 전 |
| gear_assembly | 84/100 (20 seed: 17/20) | 13/20 (20회) | 16/20 (20회) | 학습 전 |
| empty_can_selection | 56/100 (20 seed: 12/20) | 16/20 (20회) | 12/20 (20회) | 학습 전 |
| grasp_chip | 92/100 (20 seed: 19/20) | 19/20 (20회) | 19/20 (20회) | 84/99 |

- 모델 정의:
  - baseline: pi0.5 + AnyTouch2 촉각 토큰, demo 100개로 20k step 학습. HF `yrpark/openvtla-<task>`.
  - DAgger: baseline에서 시작해 demo:DAgger 데이터를 50:50으로 섞어 5k step 추가 학습.
  - DAgger + event head: DAgger에 촉각 이벤트 분류 loss(BCE, 가중치 0.1)를 더한 것. event head는 학습할 때만 쓰입니다.
  - vision-only: baseline과 같은 데이터와 설정에서 촉각 토큰만 뺀 것. HF `yrpark/pi05-vision-<task>`.
- 표의 DAgger 수치는 task별 최종 데이터로 학습한 모델입니다.
  - HF 이름 (`-<r>`): insert_USB `r1`, empty_can / grasp_chip `r2`, gear `still-r3`.
  - gear 최종 데이터는 r1+r2 데이터에서 회복 스크립트의 정지 프레임을 잘라낸 것입니다. 자르기 전 데이터로 학습한 모델은 8/20 이하였습니다.
  - 이전 라운드 결과 (참고, 모두 20회): empty_can r1 13/20 (event 14/20), gear r1 8/20 (event 5/20), gear r2 3/20 (event 6/20).
- grasp_chip vision-only: seed 10030은 시뮬레이터가 8번 재시도 모두 느린 상태에서 벗어나지 못해 제외했습니다.
  - 같은 seed로 비교하면 vision만 실패한 경우가 8회, baseline만 실패한 경우가 1회입니다 (McNemar p ≈ 0.04).
- baseline 실패 원인 (100회):
  - gear: 16회 모두 기어 삽입 후 회전 중 맞물림이 미끄러짐.
  - empty_can: 44회 중 39회가 무게를 잘못 판단해 다른 캔을 고름.
  - grasp_chip: 8회 (충돌 4, 과도하게 쥠 2, 시간 초과 2).
- 진행 중: DAgger 열을 100회로 늘리는 중입니다 (seed 10020-10099를 추가). vision-only 3개 task는 B200에서 학습이 끝나면 100회 평가합니다.

자세한 내용: [dagger.md](dagger.md), [vision_baseline.md](vision_baseline.md).

# insert_USB DAgger: 걸림(jam) 회복 데이터로 tactile 활용 학습

## 배경 (3090 서버, 2026-11-21)

시뮬레이터 평가 (insert_USB, seed 10000부터, temporal ensemble 설정):

| 모델 | 성공 |
|---|---|
| base `openvtla_insert_USB` (HF `yrpark/openvtla-insert_USB`) | 24/100 |
| ours `openvtla_insert_USB_margin` (HF `yrpark/openvtla-insert_USB-margin`) | 0/53 (중단) |

- base 실패 20개를 확인한 결과 **전부 마지막 삽입 단계**에서 실패: 잡기, 들기, 슬롯 위로 이동은 성공하고, USB가
  슬롯 입구에 어긋난 채 걸려 계속 누르다 step 한도에 도달. 이때 tactile marker가 크게 비틀림 (걸림 신호는 tactile에 있음).
- tactile 의존도 측정 (같은 프레임·같은 noise에서 tactile만 교체, 20-step chunk 팔 관절 평균 변화):
  base는 0.001~0.002 rad로 tactile을 거의 쓰지 않고, ours는 0.03~0.09 rad로 과도하게 휘둘림 (margin loss 영향으로
  추정). 정답 오차는 ours가 1.5~1.8배 나쁨.
- 학습 데이터(스크립트 시연)에는 걸림과 회복 장면이 없음 → tactile이 걸림을 표현해도 대처 행동을 배울 데이터가 없음.

## 방법

1. **DAgger 수집** (시뮬레이터 서버, `examples/openvitac/openvitac_client/policy/openpi_dagger`):
   base 정책을 그대로 실행하다 시뮬레이터의 USB–슬롯 상대 pose로 걸림을 감지하면 스크립트 전문가가 개입
   (10 mm 들기 → 슬롯 입구 위로 재정렬 → 재삽입, 스크립트 시연과 같은 primitive). 모든 프레임을 기록.
   - 걸림 조건: 삽입 구간(축에서 8 mm 이내, 삽입 위치 위 25 mm 이내)에서 (a) 플러그가 입구에 닿았는데(z < 13.9 mm)
     xy 오차 > 1.2 mm, 또는 (b) 15 policy step 동안 0.5 mm 이상 내려가지 못함. episode당 최대 2회 개입.
   - 전문가 프레임은 시연과 같이 sim 2 step마다 기록, action = 다음 프레임의 관절 상태.
2. **변환** (`examples/openvitac/convert_dagger.py`): 전문가 구간과 개입 없이 성공한 episode만 학습 데이터로 사용
   (개입이 필요했던 episode의 정책 구간은 교정 대상 행동이므로 제외). 구간마다 별도 LeRobot episode.
   AnyTouch2 특징은 rollout 전체 기준 history(4프레임, stride 2, 배경 = 첫 tactile 프레임)로 계산 후 잘라냄.
   걸림 라벨 `tactile/event_jam.npy`: 플러그가 입구 높이(z < 13.9 mm)에서 xy > 1.2 mm 또는 기울기 > 8°.
3. **학습** (B200): base 체크포인트에서 fine-tune, base의 norm stats 그대로 사용. 시연 데이터와 DAgger 데이터를
   50:50으로 샘플링 (`dataset_weights`).
   - `openvtla_insert_USB_dagger`: 행동 복제(BC)만
   - `openvtla_insert_USB_dagger_event`: + tactile event head (풀링한 tactile 토큰 → 걸림 분류, BCE λ=0.1,
     양성 가중치 5). tactile 토큰이 "걸림" 같은 굵은 패턴을 표현하도록 유도하되, margin처럼 action을 tactile 세부에
     민감하게 만드는 압박은 없음. 학습 전용, 추론은 base와 동일.
   - 5k step, batch 64, LR 2.5e-5 (warmup 200, cosine → 2.5e-6).

## 데이터 (HF `yrpark/openvitac-openvtla-data`, public)

| 경로 | 내용 |
|---|---|
| `lerobot/openvitac/insert_USB_dagger_gelsight/` | 변환된 학습 데이터 + `tactile/` (AnyTouch2 특징 GPU fp16, `event_jam.npy`, `frame_info.npz`, `anytouch2_meta.json`) |
| `raw/insert_USB_dagger_r1/*.hdf5` | 원본 rollout (재변환용: 다른 걸림 기준, 정책 구간 포함 등) |
| `lerobot/openvitac/insert_USB_gelsight/` | 기존 스크립트 시연 (변경 없음) |

수집 통계는 `lerobot/openvitac/insert_USB_dagger_gelsight/tactile/anytouch2_meta.json` 참고.

## B200에서 학습

```bash
cd tactile_base && git pull
export HF_LEROBOT_HOME=/path/to/lerobot OPENPI_DATA_HOME=/path/to/openpi_cache
# 데이터: 시연 + DAgger (이미 시연이 있으면 dagger만)
uv run huggingface-cli download yrpark/openvitac-openvtla-data --repo-type dataset \
    --include "lerobot/openvitac/insert_USB_gelsight/*" "lerobot/openvitac/insert_USB_dagger_gelsight/*" \
    --local-dir /tmp/ovt && rsync -a /tmp/ovt/lerobot/ $HF_LEROBOT_HOME/
# base 체크포인트 (params + norm stats)
uv run huggingface-cli download yrpark/openvtla-insert_USB --include "19999/params/*" "19999/assets/*" \
    "19999/_CHECKPOINT_METADATA" --local-dir checkpoints/openvtla_insert_USB/baseline
# 학습 (norm stats 계산 불필요: base 체크포인트의 assets 사용)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py openvtla_insert_USB_dagger_event --exp-name=r1 \
    --grad-accum-steps 1 --num-workers 8 --no-wandb-enabled
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py openvtla_insert_USB_dagger --exp-name=r1 \
    --grad-accum-steps 1 --num-workers 8 --no-wandb-enabled
```

- `checkpoints/openvtla_insert_USB/baseline/19999`는 상대 경로이므로 repo 루트에서 실행.
- GPU를 단독으로 쓰지 않으면 `--grad-accum-steps`를 올리고 `XLA_PYTHON_CLIENT_MEM_FRACTION`을 남은 메모리에 맞출 것
  (`B200_NOTES.md`).
- 50 step 테스트: `--num-train-steps 50 --overwrite`.

## 평가 (시뮬레이터 서버, 20 episode)

체크포인트를 3090 서버로 옮겨 서버를 띄우고 `examples/openvitac/openvitac_client/README.md`의 평가 명령으로 seed
10000부터 20개. 비교 기준은 같은 seed 20개의 base 결과: 3/20 성공 (10006, 10010, 10015). seed 10002는 스크립트
전문가도 실패하는 seed라 error로 빠지므로 20개는 seed 10000~10020.

## 다음 라운드

fine-tune한 정책으로 다시 DAgger를 수집하면(다른 seed 구간) 새 정책이 도달하는 걸림 상태에 대한 교정 데이터가 쌓임.
`deploy_insert_USB.yml`의 서버만 새 체크포인트로 바꾸면 됨.

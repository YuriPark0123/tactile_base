# B200 서버 학습 메모 (임시)

B200(192 GB, Blackwell sm_100)에서 `openvtla_<task>`를 학습할 때 확인할 점. 기준 환경은 RTX 3090 서버(README_tactile_base.md).
학습에는 시뮬레이터가 필요 없음 (녹화된 데모만 사용). 시뮬레이터(RT core)는 OpenViTac 성공률 평가에만 필요.

## 1. 버전 호환성 (확인 필요)

`uv.lock` 고정 버전과 B200 지원 여부:

| 패키지 | 고정 버전 | B200 |
|---|---|---|
| jax / jaxlib / jax-cuda12-plugin | 0.5.3 | 미검증. ptxas(nvidia-cuda-nvcc-cu12 12.9)가 sm_100을 지원하므로 동작할 가능성은 있음 |
| nvidia-cublas-cu12 / cuda-runtime | 12.6 | Blackwell 전용 커널 없음 (12.8+), 에러 또는 성능 저하 가능 |
| nvidia-cudnn-cu12 | 9.5.1 | Blackwell 정식 지원은 9.7+ |
| torch | 2.7.1 (cu126) | **동작 안 함** (지원 arch sm_50~sm_90). AnyTouch2 특징 변환에만 쓰므로 `--device cpu` 사용 |

드라이버 570.211.01 / CUDA 12.8 은 B200 지원, 문제 없음.

먼저 50 step 테스트로 확인. 에러 없이 돌고 loss가 3090과 비슷하면(3090: step 50 loss ≈ 0.18) 버전 그대로 사용.
에러가 나면 jax를 CUDA 12.8+ 빌드로 올려야 함 → 3090 서버와 JAX 버전이 달라지므로 비교 실험 시 감안.

## 2. 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env   # uv 없을 때 (관리자 권한 불필요)
git clone https://github.com/YuriPark0123/tactile_base.git && cd tactile_base
GIT_LFS_SKIP_SMUDGE=1 uv sync
git submodule update --init third_party/AnyTouch2
uv run huggingface-cli login     # AnyTouch2 가중치(xxuan01/AnyTouch2-Model)는 gated repo
```

## 3. 데이터

데이터, AnyTouch2 특징, norm stats(`assets/`)는 git에 없음. 둘 중 하나:

- **3090 서버에서 복사 (비교 실험이면 권장, 특징 해시까지 동일)**
  - `$HF_LEROBOT_HOME/openvitac/<task>_gelsight` (AnyTouch2 캐시 `tactile/` 포함)
  - `assets/openvtla_<task>/` → clone한 repo의 같은 위치
  - (선택) `$OPENPI_DATA_HOME` (pi05_base, 토크나이저). 없으면 학습 시작 시 자동 다운로드(~12 GB)
- **직접 생성**: README_tactile_base.md 의 Usage 1~3단계. 단 특징 변환은 `--device cpu --dtype float32` 필수
  (torch가 B200 미지원). CPU 변환은 느림: 3090 서버 기준 grasp_chip(43k 프레임) ≈ 3.5시간.
  3090 서버의 insert_USB / gear_assembly / empty_can_selection 특징은 GPU fp16, grasp_chip은 CPU fp32로 계산됨
  → 직접 생성하면 특징 값이 미세하게 다름 (`tactile/anytouch2_meta.json`의 device/dtype 확인).

## 4. 학습

```bash
export HF_LEROBOT_HOME=... OPENPI_DATA_HOME=...
TASK=insert_USB
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name openvtla_$TASK   # 직접 생성한 경우만

# 50 step 테스트
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/train.py openvtla_$TASK --exp-name=b200_test \
    --grad-accum-steps 2 --num-workers 8 --num-train-steps 50 --no-wandb-enabled --overwrite
```

- **GPU를 다른 작업과 공유 중이면** (확인 시점: 86.8 GB 사용, 100% util) `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`는
  전체의 95%를 선점하려다 OOM. 남은 메모리에 맞춰 0.45 등으로 낮출 것. 속도도 나눠 쓰는 만큼 느려짐.
- **batch**: 실효 batch는 64 고정(`batch_size`). `--grad-accum-steps`로 한 번에 올리는 micro-batch만 바뀜
  (4 → 16x4, 2 → 32x2, 1 → 64x1). GPU를 단독으로 쓰면 1 가능. 수학적으로 같은 batch 64이나 수치 오차 수준 차이 있음.
  3090 기준 실험과 엄밀히 맞추려면 4 유지.
- **`--num-workers`**: 기본 2. B200은 계산이 빨라 PNG 디코딩이 병목 → 8 이상 권장.
- **예상 속도**: 3090은 ~20 s/step (task당 ~4.6일). B200 단독 사용 시 1~2 s/step(task당 6~12시간)으로 추정, 미측정.
- 본 학습: `--exp-name`을 바꾸고 `--num-train-steps` 제거, `--keep-period None` (체크포인트 ~9 GB/개).

## 5. 환경 비교

```bash
uv run scripts/env_fingerprint.py > fp_b200.txt   # 3090 서버 출력과 diff
```

GPU / 드라이버 / cuDNN 줄은 당연히 다름. 패키지, pretrained 파일, 데이터 해시가 같은지 확인.

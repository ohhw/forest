# YOLO 설정 파일 작성 가이드

## 기본 구조

설정 파일은 두 가지로 구성됩니다:
- `base.yaml`: 모든 임산물에 공통적인 설정
- `{product}.yaml`: 임산물별 특화 설정

## Detection 설정 (configs/models/dod/)

### base.yaml

```yaml
# Task 타입
task: detect

# 경로 설정
paths:
  data_root: /hdd/datasets/dod_data     # 데이터셋 루트
  output_root: /home/hwoh/detection     # 출력 루트

# 공통 설정
common:
  user: hwoh
  exist_ok: true

# 기본 학습 파라미터
training:
  epochs: 500
  batch: 32
  patience: 150
  optimizer: AdamW
  imgsz: 640
  lr0: 0.001
  lrf: 0.01
  dropout: 0.2
  iou: 0.5
  plots: true

# 기본 예측 설정
prediction:
  save: true
  save_crop: true
  save_txt: true
  conf: 0.5
  iou: 0.45
```

### 임산물별 설정 (예: csn.yaml)

```yaml
# 임산물 정보
product: csn              # 임산물 코드
model: yolo11n           # 모델 크기 (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
data_version: v2         # 데이터셋 버전

# 학습 파라미터 (base.yaml 덮어쓰기)
training:
  # csn 최적 하이퍼파라미터
  epochs: 500           # base와 동일하면 생략 가능
  batch: 32
  patience: 150
  dropout: 0.24         # CSN 특화 값
  iou: 0.54
  lr0: 0.0005
  lrf: 0.00001
  optimizer: AdamW

# 예측 파라미터
prediction:
  conf: 0.5
```

**병합 결과**: `base.yaml`의 값을 기본으로 하고, `csn.yaml`의 값이 덮어씁니다.

## Classification 설정 (configs/models/cls/)

### base.yaml

```yaml
task: classify

paths:
  data_root: /hdd/datasets/cls_data
  output_root: /home/hwoh/classification

common:
  user: hwoh
  exist_ok: true

training:
  epochs: 300
  batch: 256
  patience: 50
  optimizer: AdamW
  imgsz: 224           # Classification은 224
  lr0: 0.0005
  lrf: 0.00001
  dropout: 0.3
  iou: 0.5
  plots: true

prediction:
  save: true
  save_txt: true
```

### 임산물별 설정 (예: jjb.yaml)

```yaml
product: jjb
model: yolo11s-cls      # Classification 모델
data_version: v1

training:
  epochs: 100           # JJB는 100 에폭
  batch: 128            # 배치 크기 128
  # 나머지는 base.yaml 사용

prediction:
  save: true
  save_txt: true
```

## 하이퍼파라미터 설명

### 학습 파라미터

| 파라미터 | 설명 | 추천값 |
|---------|------|--------|
| `epochs` | 학습 반복 횟수 | Detection: 300-500<br>Classification: 100-300 |
| `batch` | 배치 크기 | GPU 메모리에 따라 조정<br>16, 32, 64, 128, 256 |
| `patience` | 조기 종료 인내 | epochs의 20-30% |
| `dropout` | 드롭아웃 비율 | 0.1 ~ 0.4 |
| `iou` | IoU threshold | 0.4 ~ 0.7 |
| `lr0` | 초기 학습률 | 0.0001 ~ 0.01 |
| `lrf` | 학습률 감소 비율 | 0.00001 ~ 0.1 |
| `optimizer` | 옵티마이저 | AdamW, SGD, Adam |
| `imgsz` | 이미지 크기 | Detection: 640<br>Classification: 224 |

### 예측 파라미터

| 파라미터 | 설명 | 추천값 |
|---------|------|--------|
| `conf` | Confidence threshold | 0.25 ~ 0.7 |
| `iou` | NMS IoU threshold | 0.4 ~ 0.6 |
| `save` | 결과 이미지 저장 | true |
| `save_crop` | Crop 저장 (Detection) | true |
| `save_txt` | 라벨 txt 저장 | true |

## 데이터셋 경로 자동 구성

### Detection

설정:
```yaml
product: csn
data_version: v2
```

자동 생성 경로:
```
/hdd/datasets/dod_data/csn/csn_defect_detection_data_v2.yaml
```

### Classification

설정:
```yaml
product: csn
data_version: v9
```

자동 생성 경로:
```
/hdd/datasets/cls_data/csn/v9/
```

## 모델 이름 자동 생성

설정:
```yaml
product: csn
model: yolo11n
```

학습 시 자동 생성:
```
csn_dod_11n_25121910h
└─┬──┘│  └─┬┘└──┬──┘
  │   │    │    └─ 날짜+시간 (25년12월19일10시)
  │   │    └─ 모델 크기
  │   └─ task (dod=detect, cls=classify)
  └─ 임산물
```

## 실제 사용 예시

### 예시 1: CSN (밤) Detection 학습

**1. 설정 파일 확인** (`configs/models/dod/csn.yaml`)

```yaml
product: csn
model: yolo11n
data_version: v2

training:
  epochs: 500
  batch: 32
  patience: 150
  dropout: 0.24
  iou: 0.54
  lr0: 0.0005
  lrf: 0.00001
  optimizer: AdamW
```

**2. 학습 실행**

```bash
python train.py --config configs/models/dod/csn.yaml
```

**3. 하이퍼파라미터 튜닝 (선택사항)**

```bash
# CSN 전용 튜닝 설정 사용
python tune.py --config configs/tune/dod_tune_csn.yaml --iterations 50

# 튜닝 결과를 csn.yaml에 반영

# ⚠️ 중요: 튜닝은 최적 파라미터만 찾고, 최종 모델은 생성하지 않습니다!
#          튜닝 후 업데이트된 yaml로 재학습 필요
```

**4. 최종 모델 학습 (튜닝 후)**

```bash
# 업데이트된 파라미터로 최종 모델 학습
python train.py --config configs/models/dod/csn.yaml
```

**중간 저장 기능**
- 튜닝 중 자동으로 checkpoint 저장 (`{product}/tune_checkpoints/`)
- 오류 발생 시에도 중간 결과 복구 가능
- 튜닝 완료 시 최종 백업 자동 생성

### 예시 2: JJB (대추) Classification 학습

**1. 설정 파일** (`configs/models/cls/jjb.yaml`)

```yaml
product: jjb
model: yolo11s-cls
data_version: v1

training:
  epochs: 100
  batch: 128
  # 나머지는 base.yaml 상속
```

**2. 학습 실행**

```bash
python train.py --config configs/models/cls/jjb.yaml
```

### 예시 3: WLN (호두) 새 데이터셋으로 재학습

**1. 설정 수정** (`configs/models/dod/wln.yaml`)

```bash
# 데이터 버전만 변경
vim configs/models/dod/wln.yaml

# data_version: v1 → v2로 변경
```

**2. 학습 실행**

```bash
python train.py --config configs/models/dod/wln.yaml
```

생성되는 모델명: `wln_dod_11n_26010715h` (자동 날짜/시간 포함)

### 새 임산물 추가하기

기존 임산물(csn, jjb, wln)을 템플릿으로 사용:

```bash
# 1. 기존 설정 복사
cp configs/models/dod/csn.yaml configs/models/dod/oak.yaml  # oak: 도토리

# 2. 필수 항목만 수정
vim configs/models/dod/oak.yaml
```

```yaml
product: oak              # 임산물 코드 변경
model: yolo11n            # 모델 크기
data_version: v1          # 데이터 버전

# training 파라미터는 일단 csn 값 그대로 사용
# 학습 후 튜닝으로 최적화
```

```bash
# 3. 학습
python train.py --config configs/models/dod/oak.yaml

# 4. 튜닝 (선택)
python tune.py --config configs/tune/dod_tune.yaml --product oak --iterations 30

# ⚠️ 튜닝 완료 후 반드시 재학습 필요!
python train.py --config configs/models/dod/oak.yaml
```

## 검증 및 테스트

### 설정 검증

모든 작물 설정이 올바른지 확인:

```bash
python test_tune_configs.py
```

출력 예시:
```
✅ Detection:      통과 (csn, jjb, wln, obj)
✅ Classification: 통과 (csn, jjb, wln)
✅ 개별 설정:      통과

사용 가능한 명령어:
  python tune.py --config configs/tune/dod_tune.yaml --product csn --iterations 30
  python tune.py --config configs/tune/cls_tune.yaml --product jjb --iterations 30
  ...
```

## 팁

### 실험 관리

설정 파일을 Git으로 관리하면 모든 실험이 추적됩니다:

```bash
git add configs/models/dod/csn.yaml
git commit -m "CSN: dropout 0.22→0.24, iou 0.50→0.54 변경 (mAP 0.832→0.851)"

# 튜닝 설정도 함께 관리
git add configs/tune/dod_tune_csn.yaml
git commit -m "CSN: 튜닝 범위 좁힘 (베스트 파라미터 기준 ±0.04)"
```

### 설정 재사용

좋은 결과를 낸 설정은 다른 임산물에 적용:

```bash
# CSN 설정이 좋았다면 비슷한 특성의 임산물(WLN)에 적용
cp configs/models/dod/csn.yaml configs/models/dod/wln.yaml
vim configs/models/dod/wln.yaml  # product, data_version만 변경
```

### 주석 활용

```yaml
training:
  dropout: 0.24  # 실험123에서 최적값 확인 (2025-12-19)
  iou: 0.54      # 0.5 → 0.54로 변경하여 mAP 0.02 향상
```

## 트러블슈팅

### Q: base.yaml과 product.yaml의 값이 충돌하면?

A: product.yaml이 우선입니다. 딕셔너리는 재귀적으로 병합됩니다.

```yaml
# base.yaml
training:
  epochs: 500
  batch: 32

# csn.yaml
training:
  batch: 16  # 이 값만 변경

# 결과
training:
  epochs: 500  # base에서
  batch: 16    # csn에서 덮어씀
```

### Q: 모델 파일을 찾을 수 없다는 오류

A: 사전학습 모델은 자동 다운로드됩니다. 또는 `paths.pretrained_models` 경로에 배치하세요.

### Q: 튜닝 완료했는데 모델이 안 보여요

A: **튜닝은 최적 파라미터만 찾습니다!** 최종 모델은 직접 학습해야 합니다:

```bash
# 1. 튜닝으로 파라미터 발견 (yaml 자동 업데이트)
python tune.py --config configs/tune/dod_tune_csn.yaml --auto-update

# 2. 업데이트된 설정으로 최종 모델 학습
python train.py --config configs/models/dod/csn.yaml
```

### Q: 튜닝 중에 오류가 났어요

A: 중간 저장된 checkpoint 파일 확인:

```bash
# Checkpoint 위치
ls ~/detection/{product}/tune_checkpoints/

# 결과가 저장되어 있다면 로그 확인 가능
cat ~/detection/{product}/tune_logs/*.json
```

### Q: 설정이 반영되지 않는 것 같아요

A: YAML 문법을 확인하세요. 들여쓰기는 공백 2칸입니다.

```yaml
# 올바름
training:
  epochs: 500

# 잘못됨 (탭 사용)
training:
    epochs: 500
```

### Q: 모든 작물 설정이 제대로 되어있는지 확인하려면?

A: 검증 스크립트 실행:

```bash
python test_tune_configs.py
```

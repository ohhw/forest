# YOLO 통합 학습/추론 시스템

임산물 결함 탐지(Detection)와 색택 분류(Classification)를 위한 통합 프레임워크

---

## 빠른 시작

```bash
# 학습
python train.py --config configs/models/dod/jjb.yaml

# 예측
python predict.py --config configs/models/dod/jjb.yaml --weights best.pt --source /new/images
```

---

## 디렉토리 구조

```
yolo/
├── train.py              # 학습
├── predict.py            # 추론
├── evaluate_cls.py       # 평가 (Classification 전용)
├── tune.py               # 튜닝
├── configs/              # 설정 파일
│   ├── models/dod/      # Detection (csn.yaml, jjb.yaml, wln.yaml, obj.yaml)
│   └── models/cls/      # Classification (csn.yaml, jjb.yaml)
├── core/                 # 핵심 로직
└── utils/                # 유틸리티
```

---

## 사용법

### 1. 학습

```bash
# Detection
python train.py --config configs/models/dod/jjb.yaml    # 대추
python train.py --config configs/models/dod/csn.yaml    # 밤
python train.py --config configs/models/dod/wln.yaml    # 호두

# Classification
python train.py --config configs/models/cls/jjb.yaml    # 대추
python train.py --config configs/models/cls/csn.yaml    # 밤
python train.py --config configs/models/cls/wln.yaml    # 호두


# 학습 재개 (중단된 학습 이어하기)
python train.py --config configs/models/dod/jjb.yaml --resume
```

**학습 과정** (자동 실행):
1. 설정 로드
2. 모델 학습 (300 epochs)
3. 최종 성능 평가
4. 검증 데이터 예측 (신뢰도 포함)
5. 검증 데이터 예측 (신뢰도 제외)

**결과 위치**:
```
/home/hwoh/detection/{product}/detect/{model_name}/
├── weights/best.pt              # 학습된 모델
├── results.png                  # 학습 곡선
└── pred_xxx_val/                # 예측 결과
```

### 2. 추론 (새 이미지)

```bash
# 기본
python predict.py \
  --config configs/models/dod/jjb.yaml \
  --weights best.pt \
  --source /path/to/images

# 신뢰도 조정
python predict.py --config ... --weights best.pt --source /images --conf 0.7

# 신뢰도 포함/제외 두 버전 모두 생성
python predict.py --config ... --weights best.pt --source /images --both-conf
```

### 3. 평가 (Classification)

```bash
python evaluate_cls.py --config configs/models/cls/csn.yaml --weights best.pt
```

### 4. 튜닝

```bash
python tune.py --config configs/tune/dod_tune.yaml --product jjb --iterations 50
```

---

## 설정 수정

```yaml
# configs/models/dod/jjb.yaml

product: jjb
model: yolo11l              # 모델: yolo11n/s/m/l/x
data_version: v11           # 데이터: v9/v11

training:
  epochs: 500               # 학습 에포크 (더 많을수록 오래 걸림)
  batch: 32                 # 배치 크기 (GPU 메모리 부족시 16, 8로 줄이기)
  patience: 150             # Early stopping
  lr0: 0.001                # Learning rate

prediction:
  conf: 0.5                 # 예측 신뢰도 (오검출 많으면 0.7로 올리기)
```

---

## Detection vs Classification

| 항목 | Detection | Classification |
|------|-----------|----------------|
| 목적 | 결함 위치와 종류 찾기 | 이미지 전체를 등급으로 분류 |
| 출력 | Bounding Box + 클래스 | 클래스 라벨 |
| 용도 | 결함 검출 | 색택 판정 |
| Task | detect | classify |
| 데이터 | YAML 파일 | 디렉토리 |

---

## 기존 코드 vs 새 시스템

| 기존 | 새로운 |
|------|--------|
| `detection/jjb/train_dod_jjb_yolo11.py` | `python train.py --config configs/models/dod/jjb.yaml` |
| 코드 내 날짜/시간 수정 | 자동 생성 |
| 코드 내 epochs 수정 | YAML에서 `epochs: 500` 수정 |
| 코드 내 data_nm 수정 | YAML에서 `data_version: v11` 수정 |

---

## 자주 묻는 질문

**Q1. GPU 메모리 부족**

```yaml
training:
  batch: 16  # 32 → 16으로 줄이기
```

**Q2. 학습이 너무 오래 걸림**

```yaml
training:
  epochs: 100    # 500 → 100
  patience: 50   # 150 → 50
```

**Q3. 예측 오검출이 많음**

```bash
# 신뢰도 임계값 올리기
python predict.py ... --conf 0.7  # 기본 0.5 → 0.7
```

**Q4. best.pt 파일 위치**

```
/home/hwoh/detection/{product}/detect/{model_name}/weights/best.pt
```

**Q5. 데이터 경로 에러**

YAML 파일에서 `data_version` 확인 (v9, v11 등)

**Q6. 성능이 안 좋을 때**

1. Epochs 증가: 300 → 500
2. 모델 크기 증가: yolo11n → yolo11s → yolo11m
3. Learning rate 낮추기: lr0: 0.0001

---

## 명령어 모음

```bash
# 학습
# Detection
python train.py --config configs/models/dod/csn.yaml  # 밤
python train.py --config configs/models/dod/jjb.yaml  # 대추
python train.py --config configs/models/dod/wln.yaml  # 호두

# Classification
python train.py --config configs/models/cls/csn.yaml  # 밤
python train.py --config configs/models/cls/jjb.yaml  # 대추

# 학습 재개
python train.py --config configs/models/dod/jjb.yaml --resume

# 예측
python predict.py \
  --config configs/models/dod/jjb.yaml \
  --weights /home/hwoh/detection/jjb/detect/jjb_dod_11l_xxx/weights/best.pt \
  --source /new/images

# 평가
python evaluate.py --config configs/models/cls/csn.yaml --weights best.pt

# 튜닝
python tune.py --config configs/tune/dod_tune.yaml --product jjb --iterations 50

# 도움말
python train.py --help
python predict.py --help
```

---

## 추가 정보

- **설정 파일 작성법**: [configs/README.md](configs/README.md)
- **예제 스크립트**: `examples/` 디렉토리

문제가 있으면 `python {스크립트} --help` 로 도움말을 확인하세요.

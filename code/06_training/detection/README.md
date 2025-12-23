# Detection 학습 스크립트

## 파일 목록

### 통합 스크립트 (권장)
- **train_detection.py** - 전체 제품 통합 학습 스크립트

### 백업 (원본)
- backup/train_dod_csn_yolo11.py - 밤 탐지 (원본)
- backup/train_dod_wln_yolo11.py - 도토리 탐지 (원본)
- backup/train_dod_yolo_rev.py - 범용 탐지 (원본)

---

## 사용법

### 기본 학습
```bash
# 밤(csn) 학습
python train_detection.py --product csn --model-name csn_test --data-version v8 --train

# 건대추(jjb) 학습
python train_detection.py --product jjb --model-name jjb_test --data-version v10 --train

# 호두(wln) 학습
python train_detection.py --product wln --model-name wln_test --data-version v5 --train

# 일반 객체(obj) 학습
python train_detection.py --product obj --model-name obj_test --data-version 251015_psm --train
```

### 학습 + 검증 + 예측 (올인원)
```bash
python train_detection.py \
  --product jjb \
  --model-name jjb_dod_11s_25071510h \
  --data-version v10 \
  --train \
  --validate \
  --predict \
  --images-dir /hdd/datasets/dod_data/jjb/v10/val/images
```

### 예측만 수행
```bash
python train_detection.py \
  --product csn \
  --model-name csn_existing \
  --predict \
  --model-path /home/hwoh/detection/csn/detect/csn_test/weights/best.pt \
  --images-dir /hdd/datasets/dod_data/csn/v8/val/images
```

### 사용자 정의 YOLO 모델 크기
```bash
# YOLO11n (nano - 가장 작고 빠름)
python train_detection.py --product jjb --model-name jjb_11n --data-version v10 --yolo-model 11n --train

# YOLO11l (large - 크고 정확함)
python train_detection.py --product jjb --model-name jjb_11l --data-version v10 --yolo-model 11l --train

# YOLO11x (extra large - 가장 크고 정확함)
python train_detection.py --product jjb --model-name jjb_11x --data-version v10 --yolo-model 11x --train
```

### 사용자 정의 하이퍼파라미터
```bash
python train_detection.py \
  --product jjb \
  --model-name jjb_custom \
  --data-version v10 \
  --yolo-model 11s \
  --epochs 200 \
  --batch 16 \
  --patience 75 \
  --dropout 0.25 \
  --iou 0.35 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.00001 \
  --train
```

---

## 제품별 기본 설정

### 밤 (csn)
- epochs: 150
- batch: 32
- patience: 50
- dropout: 0.25
- iou: 0.35

### 건대추 (jjb)
- epochs: 300
- batch: 32
- patience: 100
- dropout: 0.3
- iou: 0.3
- optimizer: AdamW
- lr0: 0.0005
- lrf: 0.00001

### 호두 (wln)
- epochs: 250
- batch: 32
- patience: 100
- dropout: 0.255
- iou: 0.415

### 일반 객체 (obj)
- epochs: 200
- batch: 32
- patience: 75
- dropout: 0.3
- iou: 0.35
- optimizer: AdamW

---

## 출력 구조

```
/home/hwoh/detection/{product}/detect/
└── {model_name}/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── pred_{model_name}/          # 예측 결과 (신뢰도 표시)
    │   ├── images/
    │   ├── labels/
    │   └── crops/
    ├── pred_{model_name}_without_conf/  # 예측 결과 (신뢰도 미표시)
    └── ...
```

---

## YOLO 모델 크기 선택 가이드

| 모델 | 속도 | 정확도 | 용도 |
|-----|------|--------|------|
| 11n | 매우빠름 | 낮음 | 실시간 처리, 엣지 디바이스 |
| 11s | 빠름 | 중간 | 균형잡힌 선택 (기본값) |
| 11m | 보통 | 높음 | 정확도 중시 |
| 11l | 느림 | 매우높음 | 고정확도 필요 |
| 11x | 매우느림 | 최고 | 최고 정확도 |

---

## 고급 옵션

### 학습 파라미터
- `--epochs`: 학습 에포크 수
- `--batch`: 배치 크기
- `--patience`: 조기 종료 patience
- `--dropout`: 드롭아웃 비율
- `--iou`: IoU 임계값
- `--optimizer`: 옵티마이저 (auto, SGD, Adam, AdamW)
- `--lr0`: 초기 학습률
- `--lrf`: 최종 학습률 비율

### 예측 옵션
- `--conf`: 신뢰도 임계값 (기본: 0.5)
- `--no-show-conf`: 신뢰도 숨기기
- `--no-save-crop`: 잘라낸 객체 저장 안 함
- `--no-save-txt`: 텍스트 결과 저장 안 함

---

## 도움말
```bash
python train_detection.py --help
```

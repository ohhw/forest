# 🎓 Classification 학습 스크립트

## 📋 파일 목록

### 🆕 통합 스크립트 (권장)
- **train_classification.py** - 전체 제품 통합 학습 스크립트

### 📦 백업 (원본)
- backup/train_cls_csn_yolo11.py - 밤 분류 (원본)
- backup/train_cls_jjb_yolo11.py - 잣 분류 (원본)

---

## 🚀 사용법

### 기본 사용
```bash
# 밤(csn) 학습 - 기본 설정
python train_classification.py --product csn --model-name csn_test

# 건대추(jjb) 학습 - 기본 설정
python train_classification.py --product jjb --model-name jjb_test

# 호두(wln) 학습 - 기본 설정
python train_classification.py --product wln --model-name wln_test
```

### 학습 + 자동 평가
```bash
python train_classification.py --product csn --model-name csn_25020717h --evaluate
```

### 사용자 정의 설정
```bash
python train_classification.py \
  --product jjb \
  --model-name jjb_custom \
  --epochs 50 \
  --batch 32 \
  --yolo-model yolo11n-cls.pt \
  --optimizer AdamW \
  --lr0 0.001
```

### 평가만 수행
```bash
python train_classification.py \
  --product csn \
  --model-name csn_existing \
  --evaluate-only \
  --model-path /home/hwoh/classification/runs/classify/csn_test/weights/best.pt
```

---

## 📊 제품별 기본 설정

### 밤 (csn)
- epochs: 200
- batch: 64
- optimizer: SGD
- dropout: 0.3
- lr0: 0.003
- weight_decay: 0.01
- warmup_epochs: 10.0

### 건대추 (jjb)
- epochs: 100
- batch: 128
- optimizer: auto (기본 설정 사용)

### 호두 (wln)
- epochs: 150
- batch: 64
- optimizer: AdamW
- dropout: 0.25
- lr0: 0.001

---

## 💾 출력 구조

```
/home/hwoh/classification/runs/classify/
└── {model_name}/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── confusion_matrix_viz.png
    ├── classification_report.txt
    └── ...
```

---

## 🔧 고급 옵션

모든 YOLO 학습 파라미터를 명령행에서 오버라이드할 수 있습니다:
- `--epochs`: 학습 에포크 수
- `--batch`: 배치 크기
- `--optimizer`: 옵티마이저 (SGD, Adam, AdamW, auto)
- `--dropout`: 드롭아웃 비율
- `--lr0`: 초기 학습률
- `--weight-decay`: 가중치 감소
- `--warmup-epochs`: 워밍업 에포크
- `--patience`: 조기 종료 patience
- `--cos-lr`: Cosine 학습률 스케줄 활성화

---

## 📝 도움말
```bash
python train_classification.py --help
```

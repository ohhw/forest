# 🌲 YOLO 폴더 전체 워크플로우

## 📁 폴더 구조 및 역할

```
yolo/
│
├── 📜 실행 스크립트 (사용자가 직접 실행)
│   ├── train.py          # 학습 실행
│   ├── predict.py        # 추론 실행
│   ├── evaluate_cls.py   # 평가 실행 (Classification 전용)
│   └── tune.py           # 하이퍼파라미터 튜닝
│
├── ⚙️ configs/           # 설정 파일
│   ├── README.md         # 설정 작성 가이드
│   ├── models/
│   │   ├── dod/          # Detection 설정
│   │   │   ├── base.yaml      # Detection 공통 설정
│   │   │   ├── csn.yaml       # 밤
│   │   │   ├── jjb.yaml       # 대추
│   │   │   ├── wln.yaml       # 호두
│   │   │   └── obj.yaml       # 객체
│   │   └── cls/          # Classification 설정
│   │       ├── base.yaml      # Classification 공통 설정
│   │       ├── csn.yaml       # 밤
│   │       ├── jjb.yaml       # 대추
│   │       └── wln.yaml       # 호두
│   └── tune/             # 튜닝 설정
│       ├── dod_tune.yaml      # Detection 튜닝
│       └── cls_tune.yaml      # Classification 튜닝
│
├── 🔧 core/              # 핵심 클래스 (내부 로직)
│   ├── config.py         # ConfigLoader - 설정 파일 로드/병합
│   ├── trainer.py        # YOLOTrainer - 학습 관리
│   ├── predictor.py      # YOLOPredictor - 추론 관리
│   └── evaluator.py      # ClassificationEvaluator - 평가 관리
│
├── 🛠️ utils/             # 유틸리티 (향후 확장)
│   └── __init__.py
│
└── 📚 examples/          # 예제 스크립트
    ├── example_dod_csn.sh          # Detection 학습 예제
    ├── example_cls_csn.sh          # Classification 학습 예제
    ├── example_batch_training.sh   # 배치 학습 예제
    ├── example_tune.sh             # 튜닝 예제
    └── example_compare_tasks.sh    # Task 비교 예제
```

---

## 🔄 전체 워크플로우

### 🎯 Phase 1: 설정 준비
```
configs/models/{task}/{product}.yaml 작성
↓
base.yaml + product.yaml 병합
↓
완전한 설정 딕셔너리 생성
```

### 🔬 Phase 2: 하이퍼파라미터 튜닝 (선택적)
```bash
python tune.py --config configs/tune/dod_tune.yaml --product csn --auto-update
```
```
1. configs/tune/dod_tune.yaml 로드
   ↓
2. configs/models/dod/csn.yaml 참조
   ↓
3. Ray Tune으로 50회 실험
   ↓
4. 최적 파라미터 발견
   ↓
5. tune_logs/csn_dod_tune_20251222_153045.json 저장
   ↓
6. (--auto-update 시) csn.yaml 자동 업데이트
```

### 🏋️ Phase 3: 학습
```bash
python train.py --config configs/models/dod/csn.yaml [--resume]
```
```
train.py
↓
ConfigLoader.load(csn.yaml)
  ├── base.yaml 로드
  ├── csn.yaml 로드
  └── 병합 (csn이 base 덮어씀)
↓
YOLOTrainer(config)
  ├── 작업 디렉토리 설정 (/home/hwoh/detection/csn/)
  ├── os.chdir() 변경
  └── torch.hub 경로 설정
↓
trainer.setup_model()
  └── YOLO("yolo11n.pt") 로드
↓
trainer.train(model)
  ├── 모델명 생성 (csn_dod_11n_25121910h)
  ├── 데이터 경로 구성
  ├── model.train() 실행
  │   └── runs/csn_dod_11n_25121910h/
  │       ├── weights/best.pt ⭐
  │       └── results.png
  └── best.pt 경로 반환
↓
trainer.validate(model)
  └── 성능 메트릭 출력
↓
trainer.predict_on_validation(model, show_conf=True)
  └── runs/pred_csn_dod_11n_25121910h_val/
↓
trainer.predict_on_validation(model, show_conf=False)
  └── runs/pred_csn_dod_11n_25121910h_val_without_conf/
↓
(Classification만) Confusion Matrix + Report 자동 출력
```

**학습 완료 후 구조**:
```
/home/hwoh/detection/csn/
├── yolo11n.pt                    # 사전학습 모델
├── runs/
│   ├── csn_dod_11n_25121910h/    # 학습 결과
│   │   ├── weights/
│   │   │   ├── best.pt ⭐
│   │   │   └── last.pt
│   │   ├── results.png
│   │   └── ...
│   ├── pred_csn_dod_11n_25121910h_val/         # 예측 (conf 포함)
│   └── pred_csn_dod_11n_25121910h_val_without_conf/  # 예측 (conf 제외)
└── tune_logs/                    # 튜닝 로그
    └── csn_dod_tune_20251222_153045.json
```

### 🔍 Phase 4: 추론 (새 이미지)
```bash
python predict.py --config configs/models/dod/csn.yaml \
                  --weights detection/csn/runs/csn_dod_11n_25121910h/weights/best.pt \
                  --source /path/to/new/images \
                  --conf 0.7
```
```
predict.py
↓
ConfigLoader.load(csn.yaml)
↓
YOLOPredictor(config, weights_path)
  ├── 가중치 파일 존재 확인
  └── YOLO(best.pt) 로드
↓
predictor.predict(source, conf=0.7)
  ├── 설정 + 인자 병합
  ├── model.predict() 실행
  └── runs/pred_csn_dod_11n_25121910h_251222/
      ├── image1.jpg
      ├── image2.jpg
      └── labels/
          ├── image1.txt
          └── image2.txt
↓
results 반환
```

### 📊 Phase 5: 평가 (Classification만)
```bash
python evaluate_cls.py --config configs/models/cls/csn.yaml \
                       --weights classification/csn/runs/csn_cls_11s_25121910h/weights/best.pt
```
```
evaluate_cls.py
↓
ConfigLoader.load(csn.yaml)
↓
YOLO(best.pt) 로드
↓
ClassificationEvaluator(config, model)
↓
evaluator.evaluate()
  ├── val/ 디렉토리 이미지 수집
  ├── model.predict() 실행
  ├── Confusion Matrix 계산
  ├── Classification Report 생성
  ├── Accuracy 계산
  └── runs/evaluation/251222/
      ├── confusion_matrix.png
      └── classification_report.txt
↓
메트릭 딕셔너리 반환
```

---

## 🔗 컴포넌트 간 데이터 흐름

```
configs/models/{task}/{product}.yaml → ConfigLoader
                                           ↓
                                    설정 딕셔너리
                                           ↓
                    ┌──────────────────────┼──────────────────────┐
                    ↓                      ↓                      ↓
              YOLOTrainer            YOLOPredictor    ClassificationEvaluator
                    ↓                      ↓                      ↓
                best.pt ──────────────────┴──────────────────────┘
                    ↓                      ↓                      ↓
            학습 결과 저장            예측 결과 저장          평가 리포트 저장

configs/tune/{task}_tune.yaml → tune.py
                                    ↓
                         tune_logs/*.json
                                    ↓
                    (자동 반영: --auto-update)
                                    ↓
                configs/models/{task}/{product}.yaml
```

---

## 📊 설정 파일 계층 구조

```yaml
# base.yaml (공통 기본값)
task: detect
paths:
  data_root: /hdd/datasets/dod_data
  output_root: /home/hwoh/detection
training:
  epochs: 500
  batch: 32
  dropout: 0.2
  iou: 0.5
  ...

# product.yaml (임산물별 덮어쓰기)
product: csn
model: yolo11n
data_version: v2
training:
  dropout: 0.22    # base의 0.2 덮어씀
  iou: 0.52        # base의 0.5 덮어씀
```

**병합 결과** (ConfigLoader가 자동 처리):
```python
{
  'task': 'detect',
  'product': 'csn',
  'model': 'yolo11n',
  'data_version': 'v2',
  'paths': {
    'data_root': '/hdd/datasets/dod_data',
    'output_root': '/home/hwoh/detection'
  },
  'training': {
    'epochs': 500,      # base
    'batch': 32,        # base
    'dropout': 0.22,    # product (덮어씀)
    'iou': 0.52,        # product (덮어씀)
    ...
  }
}
```

---

## 🎮 사용자 인터페이스 (실행 스크립트)

| 스크립트 | 역할 | 핵심 클래스 | 출력 |
|---|---|---|---|
| `train.py` | 학습 실행 | ConfigLoader + YOLOTrainer | best.pt + 예측 결과 |
| `predict.py` | 추론 실행 | ConfigLoader + YOLOPredictor | 예측 이미지/라벨 |
| `evaluate_cls.py` | 평가 실행 (Classification) | ConfigLoader + ClassificationEvaluator | Confusion Matrix + Report |
| `tune.py` | 튜닝 실행 | ConfigLoader + YOLO.tune() | 튜닝 로그 + auto-update |

---

## 🔧 핵심 클래스 상세

### ConfigLoader (config.py)
**역할**: 설정 파일 로드 및 병합
```python
# 사용법
loader = ConfigLoader('configs/models/dod/csn.yaml')
config = loader.load()  # base.yaml + csn.yaml 병합 결과
```

**주요 기능**:
- base.yaml + product.yaml 자동 병합
- 딕셔너리 재귀 업데이트
- YAML 파일 로드

---

### YOLOTrainer (trainer.py)
**역할**: 모델 학습 전체 프로세스 관리

**초기화**:
```python
trainer = YOLOTrainer(config)
# - 작업 디렉토리 설정
# - os.chdir() 변경
# - torch.hub 경로 설정
```

**주요 메서드**:
1. `setup_model()` - 사전학습 모델 로드
2. `train(model, resume=False)` - 학습 실행
3. `validate(model, split='val')` - 검증 실행
4. `predict_on_validation(model, show_conf=True)` - 학습 후 자동 예측

**지원 Task**: Detection + Classification

**출력 경로**: `{output_root}/{product}/runs/{model_name}/`

---

### YOLOPredictor (predictor.py)
**역할**: 학습된 모델로 새로운 이미지 추론

**초기화**:
```python
predictor = YOLOPredictor(config, weights_path='best.pt')
# - 가중치 파일 검증
# - 모델 즉시 로드
```

**주요 메서드**:
1. `predict(source, conf=None, save=None, ...)` - 추론 실행
2. `predict_with_without_conf(source, conf=None)` - confidence on/off 비교
3. `get_validation_path()` - 검증 데이터 경로 헬퍼

**지원 Task**: Detection + Classification

**출력 경로**: `{output_root}/{product}/runs/pred_{model_name}_{date}/`

---

### ClassificationEvaluator (evaluator.py)
**역할**: Classification 모델 성능 정량 평가

**초기화**:
```python
model = YOLO('best.pt')
evaluator = ClassificationEvaluator(config, model)
# - sklearn 의존성 체크
```

**주요 메서드**:
1. `evaluate(val_dir=None, save_results=True)` - 평가 실행

**지원 Task**: **Classification만**

**출력**:
- Confusion Matrix PNG
- Classification Report TXT
- 메트릭 딕셔너리 (accuracy, precision, recall, f1)

**출력 경로**: `{output_root}/{product}/runs/evaluation/{date}/`

---

## 🔄 일반적인 작업 시나리오

### Scenario 1: 새 모델 학습 (처음부터)
```bash
# 1. 설정 파일 작성/확인
vim configs/models/dod/csn.yaml

# 2. 학습
python train.py --config configs/models/dod/csn.yaml

# 3. 새 이미지로 테스트
python predict.py --config configs/models/dod/csn.yaml \
                  --weights detection/csn/runs/csn_dod_11n_25121910h/weights/best.pt \
                  --source /new/images
```

---

### Scenario 2: 하이퍼파라미터 최적화
```bash
# 1. 튜닝 실행 (자동 업데이트)
python tune.py --config configs/tune/dod_tune.yaml --product csn --auto-update

# 2. 업데이트된 설정으로 재학습
python train.py --config configs/models/dod/csn.yaml
```

---

### Scenario 3: Classification 전체 파이프라인
```bash
# 1. 학습
python train.py --config configs/models/cls/jjb.yaml

# 2. 상세 평가
python evaluate_cls.py --config configs/models/cls/jjb.yaml \
                       --weights classification/jjb/runs/jjb_cls_11s_25121910h/weights/best.pt

# 3. 새 이미지 분류
python predict.py --config configs/models/cls/jjb.yaml \
                  --weights classification/jjb/runs/jjb_cls_11s_25121910h/weights/best.pt \
                  --source /new/images
```

---

### Scenario 4: 여러 임산물 배치 학습
```bash
# examples/example_batch_training.sh 참고
for product in csn jjb wln; do
    python train.py --config configs/models/dod/${product}.yaml
done
```

---

### Scenario 5: 학습 중단 후 재개
```bash
# 학습 재개 (last.pt에서 이어서 학습)
python train.py --config configs/models/dod/csn.yaml --resume
```

---

## 💡 핵심 설계 원칙

1. **설정 주도** - 모든 동작은 YAML 설정 파일로 제어
2. **모듈화** - core/ 클래스들은 독립적으로 재사용 가능
3. **자동화** - 경로, 이름, 디렉토리 자동 생성
4. **일관성** - Detection과 Classification 동일한 인터페이스
5. **확장성** - 새 임산물 추가 시 YAML만 작성

---

## 📌 주요 특징

### ✅ 자동화된 경로 관리
- 작업 디렉토리 자동 설정
- 모델명 자동 생성 (타임스탬프 포함)
- 출력 디렉토리 자동 생성

### ✅ 설정 병합 시스템
- base.yaml (공통) + product.yaml (특화)
- 재귀적 딕셔너리 업데이트
- 임산물별 하이퍼파라미터 관리

### ✅ 통합 인터페이스
- Detection과 Classification 동일한 사용법
- 하나의 Trainer로 두 Task 모두 지원
- 하나의 Predictor로 두 Task 모두 지원

### ✅ 학습 후 자동 평가
- 학습 완료 시 자동으로 검증 데이터 예측
- Confidence 포함/제외 두 버전 자동 생성
- Classification: Confusion Matrix 자동 출력

### ✅ 튜닝 자동 반영
- `--auto-update` 플래그로 yaml 자동 업데이트
- 튜닝 로그 JSON 형태로 누적 저장
- 백업 파일 자동 생성

---

## 🎯 디렉토리 출력 위치 정리

### Detection (CSN 예시)
```
/home/hwoh/detection/csn/
├── yolo11n.pt                                    # 사전학습 모델
├── runs/
│   ├── csn_dod_11n_25121910h/                   # 학습 결과
│   ├── pred_csn_dod_11n_25121910h_val/          # 예측 (conf O)
│   ├── pred_csn_dod_11n_25121910h_val_without_conf/  # 예측 (conf X)
│   └── pred_csn_dod_11n_25121910h_251222/       # 새 이미지 예측
└── tune_logs/
    └── csn_dod_tune_20251222_153045.json        # 튜닝 로그
```

### Classification (JJB 예시)
```
/home/hwoh/classification/jjb/
├── yolo11s-cls.pt                                # 사전학습 모델
├── runs/
│   ├── jjb_cls_11s_25121910h/                   # 학습 결과
│   ├── pred_jjb_cls_11s_25121910h_val/          # 예측 (conf O)
│   ├── pred_jjb_cls_11s_25121910h_val_without_conf/  # 예측 (conf X)
│   ├── pred_jjb_cls_11s_25121910h_251222/       # 새 이미지 예측
│   └── evaluation/
│       └── 251222/
│           ├── confusion_matrix.png              # Confusion Matrix
│           └── classification_report.txt         # 분류 리포트
└── tune_logs/
    └── jjb_cls_tune_20251222_153045.json        # 튜닝 로그
```

---

## 🚀 빠른 시작 가이드

```bash
# 1. Detection 학습
python train.py --config configs/models/dod/jjb.yaml

# 2. 추론
python predict.py --config configs/models/dod/jjb.yaml \
                  --weights detection/jjb/runs/*/weights/best.pt \
                  --source /new/images

# 3. Classification 학습
python train.py --config configs/models/cls/csn.yaml

# 4. 평가
python evaluate_cls.py --config configs/models/cls/csn.yaml \
                       --weights classification/csn/runs/*/weights/best.pt

# 5. 튜닝
python tune.py --config configs/tune/dod_tune.yaml --product wln --auto-update
```

---

이 워크플로우를 통해 **단 하나의 명령어**로 복잡한 ML 파이프라인을 실행할 수 있습니다! 🎉

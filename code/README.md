# 📦 YOLO 프로젝트 코드 저장소

임산물 AI 모델 개발을 위한 코드 모음

## 📂 폴더 구조

```
code/
├── 00_environment/          # 환경 검증 도구
├── 01_validation/           # 데이터 검증 도구
├── 02_preprocessing/        # 데이터 전처리 도구
├── 03_visualization/        # 시각화 도구
├── 04_augmentation/         # 데이터 증강 및 관리
├── 05_converter/            # 이미지 포맷 변환
├── 06_training/             # 모델 학습
│   ├── classification/      # 분류 모델
│   └── detection/           # 탐지 모델
└── utils/                   # 공통 유틸리티
```

## 🎯 빠른 시작

### 1. 환경 확인
```bash
python 00_environment/check_env.py
```

### 2. 데이터 검증
```bash
python 01_validation/check_data_integrity.py --train-txt /path/to/train.txt --check-labels
```

### 3. 데이터 전처리
```bash
python 02_preprocessing/make_txt_yolo.py --product jjb --version v10
```

### 4. 모델 학습
```bash
# 분류
python 06_training/classification/train_classification.py --product csn --model-name csn_test

# 탐지
python 06_training/detection/train_detection.py --product jjb --model-name jjb_test --data-version v10 --train
```

## 📚 상세 문서

각 폴더의 README.md 참조:
- [00_environment/README.md](00_environment/README.md)
- [01_validation/README.md](01_validation/README.md)
- [02_preprocessing/README.md](02_preprocessing/README.md)
- [03_visualization/README.md](03_visualization/README.md)
- [04_augmentation/README.md](04_augmentation/README.md)
- [05_converter/README.md](05_converter/README.md)
- [06_training/classification/README.md](06_training/classification/README.md)
- [06_training/detection/README.md](06_training/detection/README.md)
- [utils/README.md](utils/README.md)

## 🛠️ 공통 유틸리티

프로젝트 전체에서 사용하는 공통 함수들:

```python
from utils import format_time, ensure_dir, collect_images

# 시간 포맷팅
elapsed = format_time(3665)  # "1시간 1분 5초"

# 디렉토리 생성
ensure_dir("/path/to/output")

# 이미지 수집
images = collect_images("/path/to/images", recursive=True)
```

자세한 내용은 [utils/README.md](utils/README.md) 참조

## 📊 지원 제품

- **csn**: 밤
- **jjb**: 잣
- **wln**: 도토리
- **obj**: 일반 객체

## 🔧 필수 패키지

```bash
pip install ultralytics opencv-python numpy scikit-learn matplotlib seaborn pytz Pillow
```

또는

```bash
pip install -r requirements.txt
```

## 📝 버전 정보

- Version: 2.0.0
- Last Updated: 2025-12-18
- Author: hwoh

## 🤝 기여

폴더 구조와 유틸리티 함수를 활용하여 코드를 작성해주세요.

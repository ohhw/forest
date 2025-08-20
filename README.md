# 🎯 YOLO/딥러닝 환경 전문 관리 시스템 v2.0

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Ubuntu%2022.04-orange.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-brightgreen.svg)

## 📋 개요

YOLO/딥러닝 환경을 위한 종합적인 시스템 관리 도구입니다. NVIDIA GPU, CUDA, PyTorch, Ultralytics YOLO 등의 복잡한 환경 설정을 자동화하고 관리합니다.

## ✨ 주요 기능

### 🔍 **환경 진단**
- 실시간 GPU 상태 모니터링 (온도, 사용률, 메모리)
- CUDA/PyTorch 호환성 검증
- 패키지 의존성 분석
- YOLO 모델 로딩 테스트

### ⚡ **CUDA 버전 관리**
- 다중 CUDA 버전 설치 및 전환
- alternatives 시스템을 통한 버전 관리
- 자동 심볼릭 링크 설정

### 📦 **패키지 관리**
- PyTorch, Ultralytics, OpenCV 등 핵심 패키지
- 버전 호환성 자동 검증
- 충돌 해결 및 자동 복구

### 🎯 **YOLO 특화 기능**
- 모델별 권장 시스템 사양 제시
- 실시간 성능 벤치마크
- 모델 테스트 및 검증
- GPU 메모리 기반 모델 추천

### � **멀티코어 CPU 최적화** (NEW!)
- Intel MKL 및 OpenMP 환경변수 자동 설정
- OMP_NUM_THREADS, OMP_SCHEDULE, KMP_AFFINITY 최적화
- conda 환경 영구 저장으로 재부팅 후에도 설정 유지
- PyTorch 멀티스레딩 성능 극대화

### ⚖️ **실시간 CPU 부하 분산** (NEW!)
- htop 연동으로 실행 중인 프로세스 자동 감지
- taskset 기반 CPU 코어별 균등 분산
- 8코어 100% 활용으로 훈련 속도 향상
- python/conda/yolo 프로세스 우선 타겟팅

### 🕒 **KST 타임존 로깅** (NEW!)
- 모든 로그를 한국표준시(KST)로 통일
- VM 시간과 실제 시간 구분으로 운영 명확성 향상
- SESSION_ID, 백업 메타데이터 시간대 일관성

### �🔧 **고급 도구**
- 자동 진단 및 복구
- 성능 벤치마크
- 시스템 정리 및 최적화
- 전문적인 로깅 시스템

## 🚀 빠른 시작

### 사전 요구사항
- Ubuntu 22.04 LTS
- NVIDIA GPU (GTX 1060 이상 권장)
- Python 3.10+
- 최소 8GB RAM, 20GB 디스크 여유공간

### 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/ohhw/forest.git
cd forest

# 실행 권한 부여
chmod +x check_pkg.sh

# 시스템 실행
./check_pkg.sh
```

## 📊 시스템 요구사항

| 모델 | GPU 메모리 | 시스템 메모리 | 용도 |
|------|------------|---------------|------|
| YOLOv11n | 2GB | 4GB | 교육/실험 |
| YOLOv11s | 4GB | 8GB | 일반 개발 |
| YOLOv11m | 6GB | 12GB | 상용 개발 |
| YOLOv11l | 8GB | 16GB | 고성능 추론 |
| YOLOv11x | 12GB+ | 24GB+ | 대규모 훈련 |

## 🎮 지원 환경

### GPU 드라이버
- NVIDIA Driver 535+ (LTS)
- NVIDIA Driver 550+ (안정)
- NVIDIA Driver 580+ (최신, 권장)

### CUDA Toolkit
- CUDA 11.8 (레거시 지원)
- CUDA 12.1 (커뮤니티 인기)
- CUDA 12.4 (권장)
- CUDA 12.6+ (최신)

### PyTorch
- PyTorch 2.0+ with CUDA support
- torchvision 0.15+
- torchaudio 2.0+

## 📖 사용 가이드

### 1. 환경 진단
전체 시스템 상태를 종합적으로 분석합니다.

```bash
./check_pkg.sh
# 메뉴에서 "1. 환경 진단" 선택
```

### 2. CUDA 버전 관리
여러 CUDA 버전을 설치하고 전환할 수 있습니다.

```bash
# 설치된 CUDA 버전 확인
ls /usr/local/cuda-*

# alternatives를 통한 버전 전환
sudo update-alternatives --config cuda
```

### 3. 멀티코어 CPU 최적화 (NEW!)
CPU 성능을 극대화하여 YOLO 훈련 속도를 향상시킵니다.

```bash
./check_pkg.sh
# 메뉴에서 "9. 멀티코어 최적화" 선택

# 설정된 환경변수 확인
echo $OMP_NUM_THREADS
echo $OMP_SCHEDULE
echo $KMP_AFFINITY
```

### 4. 실시간 CPU 부하 분산 (NEW!)
실행 중인 프로세스를 모든 CPU 코어에 균등 분산시킵니다.

```bash
./check_pkg.sh
# 메뉴에서 "10. CPU 부하 분산" 선택

# htop으로 실시간 CPU 사용률 모니터링
htop
```

### 5. YOLO 모델 테스트
설치된 환경에서 YOLO 모델의 정상 동작을 검증합니다.

```bash
# Python 환경에서 직접 테스트
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
results = model.predict('image.jpg')
"
```

## 🔧 문제 해결

### 일반적인 문제들

**1. CUDA가 인식되지 않는 경우**
```bash
# NVIDIA 드라이버 재설치
sudo apt purge nvidia-*
sudo apt install nvidia-driver-580

# 시스템 재부팅 후 확인
nvidia-smi
```

**2. PyTorch에서 GPU를 찾지 못하는 경우**
```bash
# PyTorch 재설치 (CUDA 버전에 맞게)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. YOLO 모델 로딩 실패**
```bash
# Ultralytics 재설치
pip uninstall ultralytics
pip install ultralytics

# 캐시 정리
pip cache purge
```

**4. CPU 성능 최적화 문제** (NEW!)
```bash
# 멀티코어 설정 확인
./check_pkg.sh
# 메뉴 9번으로 멀티코어 최적화 실행

# 환경변수 수동 설정 (임시)
export OMP_NUM_THREADS=8
export OMP_SCHEDULE=dynamic
export OMP_PROC_BIND=spread
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
```

**5. CPU 부하 불균형 문제** (NEW!)
```bash
# 실시간 CPU 부하 분산 적용
./check_pkg.sh
# 메뉴 10번으로 CPU 부하 분산 실행

# 수동으로 프로세스 분산
htop  # 프로세스 PID 확인
sudo taskset -cp 0-7 [PID]  # 모든 코어에 분산
```

## 📁 프로젝트 구조

```
forest/
├── check_pkg.sh           # 메인 관리 스크립트 (멀티코어 최적화 포함)
├── README.md              # 프로젝트 문서
├── LICENSE                # 라이선스 파일
├── .gitignore             # Git 무시 파일 목록 (선택적 추적)
└── code/                  # 유틸리티 및 도구
    ├── classification/    # 분류 모델 학습 코드
    │   ├── train_cls_csn_yolo11.py
    │   └── train_cls_jjb_yolo11.py
    ├── detection/         # 탐지 모델 학습 코드
    │   ├── train_dod_csn_yolo11.py
    │   ├── train_dod_wln_yolo11.py
    │   └── train_dod_yolo_rev.py
    ├── converter/         # 데이터 변환 도구
    │   ├── img_to_arr.py
    │   └── tiff_to_bmp.py
    ├── aug_annotation.py  # 어노테이션 증강
    ├── aug_image.py       # 이미지 증강
    ├── check_env.py       # 환경 검증
    ├── make_txt_yolo.py   # YOLO 형식 라벨 생성
    ├── split_images.py    # 데이터셋 분할
    └── viz_ground_truth.py # 정답 시각화
```

## 🚀 성능 최적화 가이드

### CPU 멀티코어 최적화
```bash
# 자동 최적화 (권장)
./check_pkg.sh → 메뉴 9번

# 수동 설정
export OMP_NUM_THREADS=8
export OMP_SCHEDULE=dynamic
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
```

### 실시간 부하 분산
```bash
# 자동 분산 (권장)
./check_pkg.sh → 메뉴 10번

# htop으로 모니터링
htop
```

### 시스템 요구사항 (업데이트)
- **CPU**: 8코어 이상 권장 (멀티코어 최적화 효과 극대화)
- **GPU**: NVIDIA RTX 3060 이상 (CUDA 12.x 지원)
- **메모리**: 16GB 이상 (8코어 동시 처리 시)
- **스토리지**: NVMe SSD 권장 (빠른 데이터 로딩)

## 🤝 기여하기

1. 이 저장소를 Fork하세요
2. 새로운 기능 브랜치를 생성하세요 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/AmazingFeature`)
5. Pull Request를 열어주세요

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👨‍💻 개발자

- **ohhw** - *초기 개발 및 유지보수* - [GitHub](https://github.com/ohhw)

## 🙏 감사의 말

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 구현체
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [NVIDIA](https://developer.nvidia.com/) - CUDA 및 GPU 지원

## 📊 상태

![GitHub stars](https://img.shields.io/github/stars/ohhw/forest?style=social)
![GitHub forks](https://img.shields.io/github/forks/ohhw/forest?style=social)
![GitHub issues](https://img.shields.io/github/issues/ohhw/forest)

---

> **참고**: 이 도구는 Ubuntu 22.04 환경에서 개발 및 테스트되었습니다. 다른 배포판에서는 일부 수정이 필요할 수 있습니다.

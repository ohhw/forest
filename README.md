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

### 🔧 **고급 도구**
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

### 3. YOLO 모델 테스트
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

## 📁 프로젝트 구조

```
forest/
├── check_pkg.sh           # 메인 관리 스크립트
├── README.md              # 프로젝트 문서
├── LICENSE                # 라이선스 파일
└── docs/                  # 추가 문서
    ├── installation.md    # 설치 가이드
    ├── troubleshooting.md # 문제 해결 가이드
    └── examples/          # 사용 예제
```

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

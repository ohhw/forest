#!/bin/bash

# ============================================================================
# YOLO/딥러닝 환경 전문 관리 시스템 v2.0
# ============================================================================
# 목적: YOLO, PyTorch, CUDA 통합 환경 구성 및 관리
# 대상: Ubuntu 22.04, NVIDIA GPU 환경
# 특화: 딥러닝 모델 훈련/추론 최적화
# ============================================================================

set -euo pipefail  # 엄격한 오류 처리

# 시스템 환경 검증
if [[ $(id -u) -eq 0 ]]; then
    echo "❌ 루트 권한으로 실행하지 마세요. 보안상 위험합니다."
    exit 1
fi

# 지원 OS 확인
if [[ ! -f /etc/os-release ]] || ! grep -q "Ubuntu" /etc/os-release; then
    echo "❌ Ubuntu 환경에서만 지원됩니다."
    exit 1
fi

# 로깅 시스템 설정
readonly LOG_DIR="$HOME/.cache/yolo_env_logs"
readonly SESSION_ID="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="$LOG_DIR/yolo_env_${SESSION_ID}.log"
readonly BACKUP_DIR="$LOG_DIR/backups"

mkdir -p "$LOG_DIR" "$BACKUP_DIR"

# 로그 및 상태 관리 함수
log_message() {
    local level="$1"
    local message="$2"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # 중요한 로그는 별도 보관
    if [[ "$level" == "ERROR" || "$level" == "CRITICAL" ]]; then
        echo "[$timestamp] [$level] $message" >> "$LOG_DIR/errors.log"
    fi
}

# 시스템 상태 백업
backup_system_state() {
    local backup_file="$BACKUP_DIR/system_state_${SESSION_ID}.json"
    
    cat > "$backup_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "session_id": "$SESSION_ID",
    "nvidia_driver": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null || echo 'not_installed')",
    "cuda_version": "$(nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release \([0-9.]*\).*/\1/' || echo 'not_installed')",
    "python_version": "$(python3 --version 2>/dev/null || echo 'not_installed')",
    "conda_env": "${CONDA_DEFAULT_ENV:-system}",
    "packages": $(pip list --format=json 2>/dev/null || echo '[]')
}
EOF
    
    log_message "INFO" "시스템 상태 백업 완료: $backup_file"
}

# 색상 및 UI 설정
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly PURPLE='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# 전역 상태 변수
ISSUES_FOUND=0
CURRENT_VERSIONS_SET=1
SYSTEM_MEMORY_GB=0
GPU_MEMORY_GB=0
DISK_SPACE_GB=0

# YOLO/딥러닝 특화 설정 상수
readonly CUDA_BASE_PATH="/usr/local"
readonly NVIDIA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
readonly PYTORCH_REPO_URL="https://download.pytorch.org/whl"
readonly PRIORITY_BASE=100
readonly MIN_SYSTEM_MEMORY_GB=8
readonly MIN_GPU_MEMORY_GB=4
readonly MIN_DISK_SPACE_GB=20
readonly YOLO_MODEL_SIZES=("n" "s" "m" "l" "x")
readonly SUPPORTED_YOLO_TASKS=("detect" "segment" "classify" "pose")

# 시스템 요구사항 확인
check_system_requirements() {
    log_message "INFO" "시스템 요구사항 검증 시작"
    
    # 메모리 확인 (GB)
    SYSTEM_MEMORY_GB=$(free -g | awk 'NR==2{print $2}')
    if [[ $SYSTEM_MEMORY_GB -lt $MIN_SYSTEM_MEMORY_GB ]]; then
        print_warning "시스템 메모리가 부족합니다. 최소 ${MIN_SYSTEM_MEMORY_GB}GB 필요 (현재: ${SYSTEM_MEMORY_GB}GB)"
        ((ISSUES_FOUND++))
    fi
    
    # GPU 메모리 확인
    if command -v nvidia-smi &>/dev/null; then
        GPU_MEMORY_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
        if [[ $GPU_MEMORY_GB -lt $MIN_GPU_MEMORY_GB ]]; then
            print_warning "GPU 메모리가 부족합니다. 최소 ${MIN_GPU_MEMORY_GB}GB 필요 (현재: ${GPU_MEMORY_GB}GB)"
            ((ISSUES_FOUND++))
        fi
    fi
    
    # 디스크 공간 확인
    DISK_SPACE_GB=$(df -BG "$HOME" | awk 'NR==2 {print int($4)}')
    if [[ $DISK_SPACE_GB -lt $MIN_DISK_SPACE_GB ]]; then
        print_warning "디스크 공간이 부족합니다. 최소 ${MIN_DISK_SPACE_GB}GB 필요 (현재: ${DISK_SPACE_GB}GB)"
        ((ISSUES_FOUND++))
    fi
    
    log_message "INFO" "시스템 요구사항 검증 완료 - Memory: ${SYSTEM_MEMORY_GB}GB, GPU: ${GPU_MEMORY_GB}GB, Disk: ${DISK_SPACE_GB}GB"
}

# 동적으로 설정되는 버전 정보 (사용자 선택에 따라 설정됨)
declare -A SUPPORTED_VERSIONS=()

# YOLO 모델별 권장 설정
declare -A YOLO_MODEL_REQUIREMENTS=(
    ["yolo11n"]="2GB VRAM, 4GB RAM, 교육/실험용"
    ["yolo11s"]="4GB VRAM, 8GB RAM, 일반 개발용"
    ["yolo11m"]="6GB VRAM, 12GB RAM, 상용 개발용"
    ["yolo11l"]="8GB VRAM, 16GB RAM, 고성능 추론용"
    ["yolo11x"]="12GB+ VRAM, 24GB+ RAM, 대규모 훈련용"
)

# PyTorch 임포트명 매핑
declare -A IMPORT_NAMES=(
    ["opencv-python"]="cv2"
    ["pillow"]="PIL"
    ["numpy"]="numpy"
    ["torch"]="torch"
    ["torchvision"]="torchvision"
    ["torchaudio"]="torchaudio"
    ["ultralytics"]="ultralytics"
    ["matplotlib"]="matplotlib"
    ["seaborn"]="seaborn"
    ["tensorboard"]="tensorboard"
)

# 딥러닝 환경 검증용 추가 패키지
declare -A YOLO_ESSENTIAL_PACKAGES=(
    ["matplotlib"]="3.7.0"
    ["seaborn"]="0.12.0" 
    ["tensorboard"]="2.15.0"
    ["jupyter"]="1.0.0"
    ["ipywidgets"]="8.0.0"
)

print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${BOLD} $1${NC}"
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════════════════${NC}"
    log_message "HEADER" "$1"
}

print_info() {
    echo -e "${CYAN}💡 [INFO] $1${NC}"
    log_message "INFO" "$1"
}

print_success() {
    echo -e "${GREEN}✅ [SUCCESS] $1${NC}"
    log_message "SUCCESS" "$1"
}

print_error() {
    echo -e "${RED}❌ [ERROR] $1${NC}"
    log_message "ERROR" "$1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  [WARNING] $1${NC}"
    log_message "WARNING" "$1"
}

print_critical() {
    echo -e "${RED}${BOLD}🚨 [CRITICAL] $1${NC}"
    log_message "CRITICAL" "$1"
}

print_yolo_info() {
    echo -e "${PURPLE}🎯 [YOLO] $1${NC}"
    log_message "YOLO" "$1"
}

# YOLO 환경 검증 함수
validate_yolo_environment() {
    local validation_passed=0  # 0=성공, 1=실패
    
    # 1. CUDA 가용성 검증
    if python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "CUDA 환경 정상 - PyTorch에서 GPU 인식됨"
        
        # GPU 정보 상세 출력
        python3 -c "
import torch
print(f'GPU 개수: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')
" 2>/dev/null
    else
        print_error "CUDA 환경 문제 - PyTorch에서 GPU를 인식하지 못함"
        validation_passed=1
    fi
    
    # 2. Ultralytics 패키지 확인
    if ! python3 -c "import ultralytics" &>/dev/null; then
        print_warning "Ultralytics 패키지가 설치되지 않았습니다."
        print_info "YOLO 모델 테스트를 건너뜁니다."
        echo "  💡 설치 방법: pip install ultralytics"
        return $validation_passed
    fi
    
    # 3. YOLO 모델 로딩 테스트
    print_info "YOLO 모델 로딩 테스트 중..."
    if python3 -c "
from ultralytics import YOLO
try:
    model = YOLO('yolo11n.pt')
    print('✅ YOLO 모델 로딩 성공')
    print(f'모델 디바이스: {model.device}')
except Exception as e:
    print(f'❌ YOLO 모델 로딩 실패: {e}')
    exit(1)
" 2>/dev/null; then
        print_success "YOLO 모델 정상 로딩 확인"
    else
        print_error "YOLO 모델 로딩 실패"
        print_info "가능한 원인:"
        echo "  - 모델 파일이 없음 (yolo11n.pt)"
        echo "  - 네트워크 연결 문제"
        echo "  - Ultralytics 버전 호환성 문제"
        validation_passed=1
    fi
    
    # 4. 메모리 사용량 분석
    if command -v nvidia-smi &>/dev/null; then
        print_info "GPU 메모리 사용량 분석:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while read memory; do
            used=$(echo $memory | cut -d',' -f1)
            total=$(echo $memory | cut -d',' -f2)
            usage_percent=$((used * 100 / total))
            echo "  📊 GPU 메모리 사용률: ${usage_percent}% (${used}MB/${total}MB)"
        done
    fi
    
    return $validation_passed
}

# YOLO 모델 권장사항 출력
show_yolo_recommendations() {
    print_header "YOLO 모델별 권장 시스템 사양"
    
    echo -e "${YELLOW}현재 시스템 사양:${NC}"
    echo "  🖥️  시스템 메모리: ${SYSTEM_MEMORY_GB}GB"
    echo "  🎮 GPU 메모리: ${GPU_MEMORY_GB}GB"
    echo "  💾 디스크 여유공간: ${DISK_SPACE_GB}GB"
    echo ""
    
    echo -e "${CYAN}YOLO 모델별 권장사항:${NC}"
    for model in "${!YOLO_MODEL_REQUIREMENTS[@]}"; do
        local requirements="${YOLO_MODEL_REQUIREMENTS[$model]}"
        echo "  🎯 $model: $requirements"
    done
    
    echo ""
    echo -e "${GREEN}💡 권장 사항:${NC}"
    if [[ $GPU_MEMORY_GB -ge 12 ]]; then
        print_yolo_info "고성능 GPU 환경 - YOLOv11l, YOLOv11x 모델 사용 권장"
    elif [[ $GPU_MEMORY_GB -ge 8 ]]; then
        print_yolo_info "중급 GPU 환경 - YOLOv11m, YOLOv11l 모델 사용 권장"
    elif [[ $GPU_MEMORY_GB -ge 4 ]]; then
        print_yolo_info "기본 GPU 환경 - YOLOv11n, YOLOv11s 모델 사용 권장"
    else
        print_warning "GPU 메모리 부족 - CPU 모드 또는 경량 모델(YOLOv11n) 사용 권장"
    fi
}
# 향상된 오류 처리 및 복구 함수
handle_command_result() {
    local exit_code=$1
    local operation="$2"
    local package="$3"
    local retry_count=${4:-0}
    
    case $exit_code in
        0)
            print_success "$operation 완료: $package"
            log_message "SUCCESS" "$operation completed successfully for $package"
            return 0
            ;;
        1)
            print_error "$operation 실패: $package (일반 오류)"
            log_message "ERROR" "$operation failed for $package with exit code 1"
            
            # 자동 복구 시도
            if [[ $retry_count -lt 2 ]]; then
                print_info "자동 복구 시도 중... (시도 $((retry_count + 1))/2)"
                sleep 2
                return 2  # 재시도 신호
            fi
            return 1
            ;;
        2)
            print_error "$operation 실패: $package (권한 문제)"
            print_info "💡 해결방법: sudo 권한 확인 또는 관리자에게 문의"
            log_message "ERROR" "$operation failed for $package - permission denied"
            return 1
            ;;
        100)
            print_error "$operation 실패: $package (네트워크 오류)"
            print_info "💡 해결방법: 인터넷 연결 확인 후 재시도"
            log_message "ERROR" "$operation failed for $package - network error"
            
            # 네트워크 연결 확인
            if ping -c 1 8.8.8.8 &>/dev/null; then
                print_info "인터넷 연결은 정상입니다. 저장소 문제일 수 있습니다."
            else
                print_warning "인터넷 연결에 문제가 있습니다."
            fi
            return 1
            ;;
        *)
            print_error "$operation 실패: $package (알 수 없는 오류: $exit_code)"
            log_message "ERROR" "$operation failed for $package with unknown exit code $exit_code"
            return 1
            ;;
    esac
}

# 네트워크 연결 및 저장소 상태 확인
check_network_and_repos() {
    print_info "네트워크 및 저장소 상태 확인 중..."
    
    # 기본 인터넷 연결 확인
    if ! ping -c 1 -W 5 8.8.8.8 &>/dev/null; then
        print_error "인터넷 연결에 문제가 있습니다."
        return 1
    fi
    
    # PyTorch 저장소 확인
    if ! curl -s --head "$PYTORCH_REPO_URL/cu121/" | head -1 | grep -q "200 OK"; then
        print_warning "PyTorch 저장소 연결에 문제가 있을 수 있습니다."
    fi
    
    # NVIDIA 저장소 확인
    if ! curl -s --head "$NVIDIA_REPO_URL/" | head -1 | grep -q "200"; then
        print_warning "NVIDIA 저장소 연결에 문제가 있을 수 있습니다."
    fi
    
    print_success "네트워크 및 저장소 상태 정상"
    return 0
}

# 현재 설치된 버전을 감지하는 함수들
get_current_nvidia_driver() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1
    else
        echo ""
    fi
}

get_current_cuda_version() {
    if command -v nvcc &> /dev/null; then
        nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/'
    else
        echo ""
    fi
}

get_current_package_version() {
    local package_name="$1"
    if pip show "$package_name" &>/dev/null; then
        pip show "$package_name" 2>/dev/null | grep "Version:" | awk '{print $2}'
    else
        echo ""
    fi
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# 버전 비교 함수
compare_versions() {
    local installed="$1"
    local required="$2"
    
    # 특수 문자들을 제거하고 순수 버전 번호만 추출
    local clean_installed=$(echo "$installed" | sed 's/[+].*//g' | sed 's/[a-zA-Z].*//g')
    local clean_required=$(echo "$required" | sed 's/[+].*//g' | sed 's/[a-zA-Z].*//g')
    
    # 설치된 버전이 요구 버전과 정확히 일치하는지 확인
    if [[ "$installed" == "$required" ]]; then
        echo "EXACT"
        return
    fi
    
    # 추가 수식어가 있는지 확인 (예: +cu124, -dev, -server 등)
    if [[ "$clean_installed" == "$clean_required" && "$installed" != "$required" ]]; then
        echo "VARIANT"
        return
    fi
    
    # 버전 번호 비교 (Python의 version 모듈 사용)
    python3 -c "
import sys
from packaging import version
try:
    if version.parse('$clean_installed') > version.parse('$clean_required'):
        print('HIGHER')
    elif version.parse('$clean_installed') < version.parse('$clean_required'):
        print('LOWER')
    else:
        print('SAME')
except:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN"
}

# 버전 상태 출력 함수
print_version_status() {
    local package="$1"
    local installed="$2"
    local required="$3"
    local status="$4"
    
    case "$status" in
        "NOT_INSTALLED")
            echo -e "  ${RED}✗${NC} $package: ${RED}미설치${NC}"
            ;;
        "INSTALLED")
            echo -e "  ${GREEN}✓${NC} $package: $installed"
            ;;
        *)
            echo -e "  ${GREEN}✓${NC} $package: $installed"
            ;;
    esac
}

# NVIDIA 드라이버 옵션 (ubuntu-drivers devices 출력 기반)
declare -A NVIDIA_DRIVER_OPTIONS=(
    ["1"]="580|최신, CUDA 13.x 지원, RTX 40/30 시리즈 최적화, 권장"
    ["2"]="580-open|오픈소스 최신 버전, CUDA 13.x 지원"
    ["3"]="575|안정적, CUDA 12.x 완벽 지원, RTX 40/30 시리즈 지원"
    ["4"]="575-open|오픈소스 버전, CUDA 12.x 지원, 커뮤니티 선호"
    ["5"]="575-server|서버용, 장기 지원, 안정성 우선"
    ["6"]="575-server-open|서버용 오픈소스, 안정성과 투명성"
    ["7"]="570|안정적, CUDA 11.x/12.x 지원, 검증된 버전"
    ["8"]="570-open|오픈소스 안정 버전, CUDA 11.x/12.x 지원"
    ["9"]="570-server|서버용 안정 버전, 장기 지원"
    ["10"]="570-server-open|서버용 오픈소스 안정 버전"
    ["11"]="550|CUDA 11.x/12.x 지원, 검증된 안정 버전"
    ["12"]="550-open|오픈소스 검증 버전, CUDA 11.x/12.x 지원"
    ["13"]="535|LTS 버전, CUDA 11.x 완벽 지원, 엔터프라이즈용"
    ["14"]="535-open|LTS 오픈소스 버전"
    ["15"]="535-server|LTS 서버용, 장기 지원 보장"
    ["16"]="535-server-open|LTS 서버용 오픈소스"
)

declare -A CUDA_TOOLKIT_OPTIONS=(
    ["1"]="13.0|최신, CUDA 13.x, 차세대 GPU 지원"
    ["2"]="12.9|최신 안정, PyTorch 2.5.x, Ultralytics 8.x 완벽 지원"
    ["3"]="12.8|PyTorch 2.5.x, RTX 40/30 시리즈 최적화"
    ["4"]="12.6|PyTorch 2.4.x/2.5.x, Ultralytics 8.x"
    ["5"]="12.5|PyTorch 2.4.x, 안정적, 광범위 호환"
    ["6"]="12.4|PyTorch 2.3.x/2.5.x, 검증된 버전, 권장"
    ["7"]="12.3|PyTorch 2.3.x, RTX 40 시리즈 호환"
    ["8"]="12.2|PyTorch 2.2.x, 안정적"
    ["9"]="12.1|PyTorch 2.1.x/2.2.x, 커뮤니티 인기"
    ["10"]="12.0|PyTorch 2.0.x, 구형 호환"
    ["11"]="11.8|PyTorch 1.13.x~2.0.x, 레거시 지원"
    ["12"]="11.7|PyTorch 1.12.x~1.13.x, 구형 환경"
)

declare -A PYTORCH_OPTIONS=(
    ["1"]="2.5.1+cu121|CUDA 12.1, 최신, Ultralytics 8.3.x 완벽 호환, 권장"
    ["2"]="2.5.1+cu124|CUDA 12.4, 최신, Ultralytics 8.3.x 호환"
    ["3"]="2.4.0+cu121|CUDA 12.1, 안정적, 최신 프로젝트 다수 적용"
    ["4"]="2.3.0+cu121|CUDA 12.1, RTX 40/30 시리즈 호환"
    ["4"]="2.2.2+cu118|CUDA 11.8, 커뮤니티에서 가장 많이 사용, 안정적"
    ["5"]="1.13.1+cu117|CUDA 11.7, 구형 환경, 레거시 코드 호환"
)

declare -A TORCHVISION_OPTIONS=(
    ["1"]="0.18.0|PyTorch 2.5.x 전용, 최신"
    ["2"]="0.17.0|PyTorch 2.4.x 전용, 안정적"
    ["3"]="0.16.0|PyTorch 2.3.x 전용, RTX 40/30 시리즈 호환"
    ["4"]="0.15.2|PyTorch 2.2.x 전용, 커뮤니티 인기"
    ["5"]="0.14.1|PyTorch 1.13.x 전용, 구형 환경"
)

declare -A TORCHAUDIO_OPTIONS=(
    ["1"]="2.5.1|PyTorch 2.5.x 전용, 최신"
    ["2"]="2.4.0|PyTorch 2.4.x 전용, 안정적"
    ["3"]="2.3.0|PyTorch 2.3.x 전용, RTX 40/30 시리즈 호환"
    ["4"]="2.2.2|PyTorch 2.2.x 전용, 커뮤니티 인기"
    ["5"]="0.13.1|PyTorch 1.13.x 전용, 구형 환경"
)

declare -A ULTRALYTICS_OPTIONS=(
    ["1"]="8.3.62|최신, PyTorch 2.5.x~2.3.x, CUDA 12.x~11.x 완벽 호환"
    ["2"]="8.0.20|PyTorch 2.0~2.2, CUDA 11.x, 안정적, 커뮤니티 인기"
    ["3"]="8.0.4|PyTorch 1.13.x, CUDA 11.x, 구형 환경 호환"
)

declare -A NUMPY_OPTIONS=(
    ["1"]="1.26.4|최신, PyTorch/Ultralytics 최신 버전 호환"
    ["2"]="1.24.4|안정적, PyTorch 2.x, Ultralytics 8.x 호환"
    ["3"]="1.23.5|구형 환경, PyTorch 1.x 호환"
)

declare -A OPENCV_OPTIONS=(
    ["1"]="4.9.0.80|최신, PyTorch/Ultralytics 최신 버전 호환"
    ["2"]="4.8.0.76|안정적, 커뮤니티 인기"
    ["3"]="4.7.0.72|구형 환경, PyTorch 1.x 호환"
)

declare -A PILLOW_OPTIONS=(
    ["1"]="10.4.0|최신, PyTorch/Ultralytics 최신 버전 호환"
    ["2"]="9.5.0|안정적, 커뮤니티 인기"
    ["3"]="8.4.0|구형 환경, PyTorch 1.x 호환"
)

# 동적으로 설정되는 필수 버전 정보 (사용자 선택에 따라 설정됨)
REQUIRED_NVIDIA_DRIVER=""
REQUIRED_CUDA_TOOLKIT=""

# 환경 진단 함수 (개선된 버전 비교 포함)
# 환경 진단 함수 (YOLO 특화 개선 버전)
function diagnose_environment() {
    clear
    print_header "🔍 YOLO/딥러닝 환경 종합 진단"
    
    # 시스템 상태 백업
    backup_system_state
    
    # 시스템 요구사항 확인
    check_system_requirements
    
    # 네트워크 상태 확인
    check_network_and_repos
    
    local issues_count=0
    
    # 1. 시스템 정보 섹션
    echo ""
    echo -e "${CYAN}${BOLD}📊 시스템 정보${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "  🖥️  OS: $(lsb_release -d | cut -f2)"
    echo "  💾 시스템 메모리: ${SYSTEM_MEMORY_GB}GB"
    echo "  💿 디스크 여유공간: ${DISK_SPACE_GB}GB"
    echo "  🐍 Python: $(python3 --version 2>/dev/null || echo '미설치')"
    echo "  📦 Conda 환경: ${CONDA_DEFAULT_ENV:-"기본 환경"}"
    
    # 2. NVIDIA/CUDA 환경 섹션
    echo ""
    echo -e "${CYAN}${BOLD}🎮 NVIDIA/CUDA 환경${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # NVIDIA 드라이버 상세 정보
    if command -v nvidia-smi &>/dev/null; then
        local driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        
        echo -e "  ${GREEN}✅ NVIDIA 드라이버: v$driver_ver${NC}"
        echo -e "  ${GREEN}✅ GPU: $gpu_name${NC}"
        echo -e "  ${GREEN}✅ GPU 메모리: ${gpu_memory}MB ($(($gpu_memory / 1024))GB)${NC}"
        
        # GPU 온도 및 사용률
        local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        echo "  🌡️  GPU 온도: ${gpu_temp}°C"
        echo "  📈 GPU 사용률: ${gpu_util}%"
        
        # CUDA 호환성 확인
        if [[ -n "$REQUIRED_NVIDIA_DRIVER" ]]; then
            local driver_status=$(compare_versions "$driver_ver" "$REQUIRED_NVIDIA_DRIVER")
            if [[ "$driver_status" != "LOWER" ]]; then
                echo -e "  ${GREEN}✅ 드라이버 버전 호환성: 양호${NC}"
            else
                echo -e "  ${YELLOW}⚠️  드라이버 업데이트 권장 (목표: v$REQUIRED_NVIDIA_DRIVER)${NC}"
                ((issues_count++))
            fi
        fi
    else
        echo -e "  ${RED}❌ NVIDIA 드라이버: 미설치${NC}"
        ((issues_count++))
    fi
    
    # CUDA Toolkit 상세 정보
    if command -v nvcc &>/dev/null; then
        local cuda_ver=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        echo -e "  ${GREEN}✅ CUDA Toolkit: v$cuda_ver${NC}"
        
        # 다중 CUDA 설치 확인
        local cuda_installations=$(find /usr/local -maxdepth 1 -name "cuda-*" -type d 2>/dev/null | wc -l)
        if [[ $cuda_installations -gt 1 ]]; then
            echo "  📚 다중 CUDA 버전 설치: ${cuda_installations}개"
            find /usr/local -maxdepth 1 -name "cuda-*" -type d | sort | while read cuda_dir; do
                local ver=$(basename "$cuda_dir" | sed 's/cuda-//')
                echo "    - CUDA $ver"
            done
        fi
        
        if [[ -n "$REQUIRED_CUDA_TOOLKIT" ]]; then
            local cuda_status=$(compare_versions "$cuda_ver" "$REQUIRED_CUDA_TOOLKIT")
            if [[ "$cuda_status" != "LOWER" ]]; then
                echo -e "  ${GREEN}✅ CUDA 버전 호환성: 양호${NC}"
            else
                echo -e "  ${YELLOW}⚠️  CUDA 업데이트 권장 (목표: v$REQUIRED_CUDA_TOOLKIT)${NC}"
                ((issues_count++))
            fi
        fi
    else
        echo -e "  ${RED}❌ CUDA Toolkit: 미설치${NC}"
        ((issues_count++))
    fi
    
    # 3. Python/PyTorch 환경 섹션
    echo ""
    echo -e "${CYAN}${BOLD}🐍 Python/PyTorch 환경${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # PyTorch CUDA 연동 확인
    if python3 -c "import torch" &>/dev/null; then
        local torch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        local cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        local cuda_device_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        
        echo -e "  ${GREEN}✅ PyTorch: v$torch_version${NC}"
        
        if [[ "$cuda_available" == "True" ]]; then
            echo -e "  ${GREEN}✅ PyTorch CUDA 지원: 활성화 (GPU ${cuda_device_count}개)${NC}"
            
            # GPU 디바이스 정보
            python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(f'    🎮 GPU {i}: {device_name}')
" 2>/dev/null
        else
            echo -e "  ${RED}❌ PyTorch CUDA 지원: 비활성화${NC}"
            echo -e "    ${YELLOW}💡 CPU 모드에서만 동작합니다${NC}"
            ((issues_count++))
        fi
    else
        echo -e "  ${RED}❌ PyTorch: 미설치${NC}"
        ((issues_count++))
    fi
    
    # 핵심 패키지 상태 확인
    local core_packages=("torch" "torchvision" "torchaudio" "ultralytics" "numpy" "opencv-python" "pillow")
    local optional_packages=("matplotlib" "seaborn" "tensorboard" "jupyter")
    
    echo ""
    echo -e "  ${BOLD}핵심 패키지:${NC}"
    for pkg in "${core_packages[@]}"; do
        local import_name="${IMPORT_NAMES[$pkg]:-$pkg}"
        local current_version=$(python3 -c "
try:
    import $import_name
    if hasattr($import_name, '__version__'):
        print($import_name.__version__)
    else:
        import pkg_resources
        print(pkg_resources.get_distribution('$pkg').version)
except:
    print('NOT_INSTALLED')
" 2>/dev/null)
        
        if [[ "$current_version" == "NOT_INSTALLED" ]]; then
            echo -e "    ${RED}❌ $pkg: 미설치${NC}"
            ((issues_count++))
        else
            # 버전 호환성 확인
            if [[ -n "${SUPPORTED_VERSIONS[$pkg]:-}" ]]; then
                local status=$(compare_versions "$current_version" "${SUPPORTED_VERSIONS[$pkg]}")
                case "$status" in
                    "EXACT"|"SAME"|"HIGHER"|"VARIANT")
                        echo -e "    ${GREEN}✅ $pkg: $current_version${NC}"
                        ;;
                    "LOWER")
                        echo -e "    ${YELLOW}⚠️  $pkg: $current_version (업데이트 권장: ${SUPPORTED_VERSIONS[$pkg]})${NC}"
                        ;;
                    *)
                        echo -e "    ${GREEN}✅ $pkg: $current_version${NC}"
                        ;;
                esac
            else
                echo -e "    ${GREEN}✅ $pkg: $current_version${NC}"
            fi
        fi
    done
    
    echo ""
    echo -e "  ${BOLD}선택적 패키지:${NC}"
    for pkg in "${optional_packages[@]}"; do
        local import_name="${IMPORT_NAMES[$pkg]:-$pkg}"
        local current_version=$(python3 -c "
try:
    import $import_name
    if hasattr($import_name, '__version__'):
        print($import_name.__version__)
    else:
        import pkg_resources
        print(pkg_resources.get_distribution('$pkg').version)
except:
    print('NOT_INSTALLED')
" 2>/dev/null)
        
        if [[ "$current_version" == "NOT_INSTALLED" ]]; then
            echo -e "    ${YELLOW}⚪ $pkg: 미설치 (선택사항)${NC}"
        else
            echo -e "    ${GREEN}✅ $pkg: $current_version${NC}"
        fi
    done
    
    
    # 4. YOLO 환경 검증
    echo ""
    echo -e "${CYAN}${BOLD}🎯 YOLO 환경 검증${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if validate_yolo_environment; then
        echo -e "  ${GREEN}✅ YOLO 환경 검증 통과${NC}"
    else
        echo -e "  ${RED}❌ YOLO 환경 검증 실패${NC}"
        ((issues_count++))
    fi
    
    # 5. 종합 결과 및 권장사항
    echo ""
    echo -e "${CYAN}${BOLD}📋 진단 결과 요약${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [[ $issues_count -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}🎉 환경 진단 완료: 모든 구성 요소가 정상입니다!${NC}"
        echo ""
        echo -e "${GREEN}✅ YOLO/딥러닝 환경이 완벽하게 구성되었습니다.${NC}"
        echo -e "${GREEN}✅ 모든 핵심 패키지가 올바르게 설치되어 있습니다.${NC}"
        echo -e "${GREEN}✅ GPU 가속이 정상적으로 작동합니다.${NC}"
        
        # YOLO 모델 권장사항 표시
        show_yolo_recommendations
        
    else
        echo -e "${YELLOW}${BOLD}⚠️  환경 진단 완료: ${issues_count}개의 문제가 발견되었습니다.${NC}"
        echo ""
        echo -e "${YELLOW}🔧 권장 해결 방법:${NC}"
        
        if [[ $issues_count -le 3 ]]; then
            echo -e "  ${CYAN}💡 '4. 개별 패키지 설치' 메뉴를 사용하여 필요한 구성 요소를 설치하세요.${NC}"
        else
            echo -e "  ${CYAN}💡 '3. 패키지 버전 구성' 메뉴에서 버전을 설정한 후 일괄 설치를 권장합니다.${NC}"
        fi
        
        echo -e "  ${CYAN}💡 설치 후 다시 진단을 실행하여 문제가 해결되었는지 확인하세요.${NC}"
        
        # 심각한 문제가 있는 경우 추가 안내
        if ! command -v nvidia-smi &>/dev/null; then
            echo ""
            echo -e "${RED}🚨 중요: NVIDIA 드라이버가 설치되지 않았습니다.${NC}"
            echo -e "${YELLOW}   GPU 가속을 사용하려면 먼저 NVIDIA 드라이버를 설치해야 합니다.${NC}"
        fi
    fi
    
    # 로그 파일 위치 안내
    echo ""
    echo -e "${CYAN}📄 자세한 로그: $LOG_FILE${NC}"
    echo -e "${CYAN}🔧 백업 파일: $BACKUP_DIR${NC}"
    
    # 다음 단계 제안
    echo ""
    echo -e "${BOLD}🚀 다음 단계 제안:${NC}"
    if [[ $issues_count -eq 0 ]]; then
        echo "  1. YOLO 모델 테스트: python3 -c \"from ultralytics import YOLO; YOLO('yolo11n.pt').predict('image.jpg')\""
        echo "  2. 모델 훈련 시작: 데이터셋 준비 후 훈련 스크립트 실행"
        echo "  3. 성능 모니터링: nvidia-smi로 GPU 사용률 확인"
    else
        echo "  1. 문제가 발견된 구성 요소부터 설치"
        echo "  2. 설치 완료 후 환경 진단 재실행"
        echo "  3. 모든 문제 해결 후 YOLO 모델 테스트"
    fi
    
    return $issues_count
}

# 나머지 함수들은 기존과 동일하게 유지...
function remove_all_packages() {
    print_header "모든 관련 패키지 삭제 (초기 환경으로 복원)"
    echo -e "${RED}⚠️  위험한 작업입니다! ⚠️${NC}"
    echo -e "${YELLOW}시스템에 설치된 모든 관련 NVIDIA 드라이버, CUDA Toolkit, 핵심 Python 패키지를 삭제합니다.${NC}"
    echo -e "${YELLOW}이 작업은 되돌릴 수 없으며, 전체 딥러닝 환경이 초기화됩니다.${NC}"
    echo ""
    echo -e "${CYAN}계속하려면 정확히 '${RED}remove${CYAN}'를 입력하세요:${NC}"
    read -p "> " confirm_keyword
    
    if [[ "$confirm_keyword" != "remove" ]]; then
        print_error "삭제가 취소되었습니다."
        echo ""
        read -p "계속하려면 Enter..."
        return
    fi
    
    echo ""
    echo -e "${RED}최종 확인: 정말로 모든 패키지를 삭제하시겠습니까? (y/N):${NC}"
    read -p "> " final_confirm
    
    if [[ "$final_confirm" != "y" && "$final_confirm" != "Y" ]]; then
        print_error "삭제가 취소되었습니다."
        echo ""
        read -p "계속하려면 Enter..."
        return
    fi
    
    echo ""
    echo -e "${GREEN}패키지 삭제를 시작합니다...${NC}"
    
    # pip 캐시 초기화
    echo -e "${CYAN}pip 캐시 초기화 중...${NC}"
    pip cache purge
    
    # NVIDIA 드라이버 전체 삭제 (설치된 모든 드라이버)
    echo -e "${CYAN}NVIDIA 드라이버 삭제 중...${NC}"
    for pkg in $(dpkg -l | grep nvidia-driver | awk '{print $2}'); do
        sudo apt-get remove --purge -y "$pkg"
    done
    
    # CUDA Toolkit 전체 삭제 (설치된 모든 toolkit)
    echo -e "${CYAN}CUDA Toolkit 삭제 중...${NC}"
    for pkg in $(dpkg -l | grep cuda-toolkit | awk '{print $2}'); do
        sudo apt-get remove --purge -y "$pkg"
    done
    
    # 추가 NVIDIA/CUDA 관련 패키지 삭제
    echo -e "${CYAN}추가 NVIDIA/CUDA 관련 패키지 삭제 중...${NC}"
    sudo apt-get autoremove --purge -y nvidia* cuda*
    
    # Python 패키지 전체 삭제 (핵심 패키지)
    echo -e "${CYAN}Python 패키지 삭제 중...${NC}"
    pip uninstall -y numpy opencv-python pillow torch torchvision torchaudio ultralytics
    
    # pip 캐시 다시 한번 정리
    pip cache purge
    
    # 환경 변수 초기화
    CURRENT_VERSIONS_SET=0
    
    print_success "모든 관련 패키지 삭제 완료! 초기 환경으로 복원되었습니다."
    echo -e "${GREEN}이제 원하는 버전을 선택해 설치할 수 있습니다.${NC}"
    echo ""
    read -p "계속하려면 Enter..."
}

# CUDA 버전 관리 함수
function manage_cuda_versions() {
    clear
    print_header "CUDA 버전 관리"
    echo -e "${CYAN}설치된 CUDA 버전들을 관리합니다.${NC}"
    echo ""
    
    # 현재 설치된 CUDA 버전들 확인
    echo -e "${YELLOW}설치된 CUDA 버전:${NC}"
    if ls ${CUDA_BASE_PATH}/cuda-* &>/dev/null; then
        for cuda_dir in ${CUDA_BASE_PATH}/cuda-*; do
            if [[ -d "$cuda_dir" ]]; then
                version=$(basename "$cuda_dir" | sed 's/cuda-//')
                echo "  - CUDA $version"
            fi
        done
    else
        echo "  설치된 CUDA 버전이 없습니다."
    fi
    echo ""
    
    # 현재 활성화된 버전
    echo -e "${YELLOW}현재 활성화된 버전:${NC}"
    local current_cuda=$(get_current_cuda_version)
    if [[ -n "$current_cuda" ]]; then
        echo "  - CUDA $current_cuda"
    else
        echo "  활성화된 CUDA가 없습니다."
    fi
    echo ""
    
    echo "1. CUDA 버전 전환"
    echo "2. CUDA 버전 추가 설치"
    echo "3. alternatives 상태 확인"
    echo "4. 설치된 CUDA 버전 상세 정보"
    echo "0. 메인 메뉴로 돌아가기"
    echo ""
    
    read -p "선택 (0-4): " cuda_choice
    case $cuda_choice in
        1)
            switch_cuda_version
            ;;
        2)
            install_additional_cuda
            ;;
        3)
            check_cuda_alternatives
            ;;
        4)
            show_cuda_details
            ;;
        0)
            return
            ;;
        *)
            print_error "잘못된 선택입니다."
            sleep 1
            manage_cuda_versions
            ;;
    esac
}

# CUDA 버전 전환 함수
function switch_cuda_version() {
    clear
    print_header "CUDA 버전 전환"
    echo -e "${CYAN}설치된 CUDA 버전 중에서 선택합니다.${NC}"
    echo ""
    
    # alternatives 설정 확인 및 실행
    if command -v update-alternatives &>/dev/null; then
        if update-alternatives --list cuda &>/dev/null; then
            echo -e "${GREEN}설치된 CUDA 버전들:${NC}"
            sudo update-alternatives --config cuda
        else
            print_warning "CUDA alternatives가 설정되지 않았습니다. 설정을 진행합니다."
            setup_cuda_alternatives
        fi
    else
        print_error "update-alternatives 명령어를 찾을 수 없습니다."
    fi
    
    echo ""
    read -p "계속하려면 Enter..."
    manage_cuda_versions
}

# CUDA alternatives 설정 함수
function setup_cuda_alternatives() {
    echo -e "${CYAN}CUDA alternatives 설정 중...${NC}"
    
    # 기존 alternatives 제거
    sudo update-alternatives --remove-all cuda 2>/dev/null || true
    
    # 설치된 CUDA 버전들을 alternatives에 등록
    local priority=$PRIORITY_BASE
    for cuda_dir in ${CUDA_BASE_PATH}/cuda-*; do
        if [[ -d "$cuda_dir" ]]; then
            version=$(basename "$cuda_dir" | sed 's/cuda-//')
            version_priority=$((${version//.}0 + priority))
            echo "  CUDA $version 등록 (우선순위: $version_priority)"
            sudo update-alternatives --install ${CUDA_BASE_PATH}/cuda cuda "$cuda_dir" "$version_priority"
            ((priority++))
        fi
    done
    
    print_success "CUDA alternatives 설정 완료!"
}

# CUDA alternatives 상태 확인 함수
function check_cuda_alternatives() {
    clear
    print_header "CUDA Alternatives 상태"
    echo ""
    
    if update-alternatives --display cuda &>/dev/null; then
        update-alternatives --display cuda
    else
        print_warning "CUDA alternatives가 설정되지 않았습니다."
        echo ""
        read -p "지금 설정하시겠습니까? (y/n): " setup_confirm
        if [[ "$setup_confirm" == "y" || "$setup_confirm" == "Y" ]]; then
            setup_cuda_alternatives
        fi
    fi
    
    echo ""
    read -p "계속하려면 Enter..."
    manage_cuda_versions
}

# 추가 CUDA 버전 설치 함수
function install_additional_cuda() {
    clear
    print_header "CUDA 버전 추가 설치"
    echo -e "${CYAN}새로운 CUDA 버전을 설치합니다.${NC}"
    echo ""
    
    echo "설치 가능한 CUDA 버전:"
    for i in "${!CUDA_TOOLKIT_OPTIONS[@]}"; do
        IFS='|' read -r version desc <<< "${CUDA_TOOLKIT_OPTIONS[$i]}"
        printf "%2s. CUDA %-8s - %s\n" "$i" "$version" "$desc"
    done
    echo ""
    
    read -p "설치할 CUDA 버전 번호 (0: 취소): " version_choice
    
    if [[ "$version_choice" == "0" ]]; then
        return
    fi
    
    if [[ -n "${CUDA_TOOLKIT_OPTIONS[$version_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${CUDA_TOOLKIT_OPTIONS[$version_choice]}"
        
        # 이미 설치되어 있는지 확인
        if [[ -d "/usr/local/cuda-$version" ]]; then
            print_warning "CUDA $version 이미 설치되어 있습니다."
        else
            echo -e "${GREEN}CUDA $version 설치를 시작합니다...${NC}"
            install_cuda_toolkit_version "$version"
            
            # alternatives에 추가
            version_priority=$((${version//.}0 + 100))
            sudo update-alternatives --install /usr/local/cuda cuda "/usr/local/cuda-$version" "$version_priority"
            print_success "CUDA $version 설치 및 alternatives 등록 완료!"
        fi
    else
        print_error "잘못된 선택입니다."
    fi
    
    echo ""
    read -p "계속하려면 Enter..."
    manage_cuda_versions
}

# CUDA 버전별 상세 정보 함수
function show_cuda_details() {
    clear
    print_header "CUDA 버전 상세 정보"
    echo ""
    
    for cuda_dir in /usr/local/cuda-*; do
        if [[ -d "$cuda_dir" ]]; then
            version=$(basename "$cuda_dir" | sed 's/cuda-//')
            echo -e "${YELLOW}CUDA $version:${NC}"
            echo "  경로: $cuda_dir"
            
            # nvcc 버전 확인
            if [[ -f "$cuda_dir/bin/nvcc" ]]; then
                nvcc_version=$("$cuda_dir/bin/nvcc" --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
                echo "  nvcc 버전: $nvcc_version"
            fi
            
            # 디스크 사용량
            if command -v du &>/dev/null; then
                size=$(du -sh "$cuda_dir" 2>/dev/null | cut -f1)
                echo "  디스크 사용량: $size"
            fi
            echo ""
        fi
    done
    
    read -p "계속하려면 Enter..."
    manage_cuda_versions
}

# 환경 버전 선택 함수
function show_menu() {
    clear
    print_header "🎯 YOLO/딥러닝 환경 전문 관리 시스템 v2.0"
    
    # 시스템 정보 간단 표시
    local current_time=$(date '+%Y-%m-%d %H:%M:%S')
    local gpu_status="❌ 미설치"
    local cuda_status="❌ 미설치"
    local pytorch_status="❌ 미설치"
    
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        gpu_status="✅ ${gpu_name}"
    fi
    
    if command -v nvcc &>/dev/null; then
        local cuda_ver=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        cuda_status="✅ CUDA v$cuda_ver"
    fi
    
    if python3 -c "import torch" &>/dev/null; then
        local torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        local cuda_available=$(python3 -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
        pytorch_status="✅ PyTorch v$torch_ver ($cuda_available)"
    fi
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}📊 현재 환경 상태${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "🖥️  시스템: Ubuntu $(lsb_release -rs) | � Python: $(python3 --version | cut -d' ' -f2)"
    echo -e "🎮 GPU: $gpu_status"
    echo -e "⚡ CUDA: $cuda_status"
    echo -e "🔥 PyTorch: $pytorch_status"
    echo -e "📅 시간: $current_time | 📝 세션: $SESSION_ID"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${BOLD}🔧 주요 기능${NC}"
    echo -e "${GREEN} 1.${NC} 🔍 환경 진단          - 전체 YOLO/딥러닝 환경 상태 분석"
    echo -e "${GREEN} 2.${NC} ⚡ CUDA 버전 관리      - 다중 CUDA 설치 및 전환"
    echo -e "${GREEN} 3.${NC} 📦 패키지 버전 구성    - 호환 버전 선택 및 설정"
    echo -e "${GREEN} 4.${NC} 🛠️  개별 패키지 설치   - 선택적 구성 요소 설치"
    echo -e "${GREEN} 5.${NC} 🗑️  환경 초기화       - 모든 관련 패키지 제거"
    echo ""
    echo -e "${BOLD}📚 고급 기능${NC}"
    echo -e "${GREEN} 6.${NC} 🎯 YOLO 모델 테스트    - 설치된 환경에서 YOLO 동작 검증"
    echo -e "${GREEN} 7.${NC} 📊 성능 벤치마크      - GPU/CPU 성능 측정"
    echo -e "${GREEN} 8.${NC} 🔧 문제 해결 도구     - 자동 진단 및 복구"
    echo -e "${GREEN} 9.${NC} 🚀 멀티코어 최적화    - CPU 코어 활용 최적화"
    echo ""
    echo -e "${RED} 0.${NC} 🚪 종료"
    echo ""
    echo -e "${CYAN}💾 로그: $LOG_FILE${NC}"
    echo -e "${CYAN}📂 백업: $BACKUP_DIR${NC}"
    echo ""
    
    read -p "선택 (0-9): " choice
    case $choice in
        1)
            diagnose_environment
            echo ""
            read -p "계속하려면 Enter..."
            show_menu
            ;;
        2)
            manage_cuda_versions
            show_menu
            ;;
        3)
            select_versions
            show_menu
            ;;
        4)
            selective_install
            show_menu
            ;;
        5)
            remove_all_packages
            show_menu
            ;;
        6)
            test_yolo_models
            show_menu
            ;;
        7)
            run_performance_benchmark
            show_menu
            ;;
        8)
            run_diagnostic_tools
            show_menu
            ;;
        9)
            optimize_multicore
            echo ""
            echo -e "${CYAN}계속하려면 Enter를 누르세요...${NC}"
            if read -t 30; then
                show_menu
            else
                echo -e "${YELLOW}시간 초과 또는 입력 오류로 메뉴로 돌아갑니다.${NC}"
                show_menu
            fi
            ;;
        0)
            print_success "🎯 YOLO 환경 관리 시스템을 종료합니다."
            echo -e "${CYAN}다음 세션에서 다시 만나요! 🚀${NC}"
            exit 0
            ;;
        *)
            print_error "잘못된 선택입니다."
            sleep 1
            show_menu
            ;;
    esac
}

# YOLO 모델 테스트 함수
test_yolo_models() {
    clear
    print_header "🎯 YOLO 모델 테스트 및 검증"
    
    print_info "YOLO 환경 검증을 시작합니다..."
    
    # 사전 요구사항 확인
    if ! python3 -c "import torch, ultralytics" &>/dev/null; then
        print_error "PyTorch 또는 Ultralytics가 설치되지 않았습니다."
        echo ""
        read -p "계속하려면 Enter..."
        return 1
    fi
    
    echo ""
    echo -e "${YELLOW}테스트할 모델을 선택하세요:${NC}"
    echo "1. YOLOv11n (Nano) - 경량, 빠른 추론"
    echo "2. YOLOv11s (Small) - 균형잡힌 성능"
    echo "3. YOLOv11m (Medium) - 높은 정확도"
    echo "4. YOLOv11l (Large) - 매우 높은 정확도"
    echo "5. YOLOv11x (Extra Large) - 최고 정확도"
    echo "6. 전체 모델 테스트"
    echo "0. 메인 메뉴로 돌아가기"
    echo ""
    
    read -p "선택 (0-6): " model_choice
    
    case $model_choice in
        1) test_single_yolo_model "yolo11n" ;;
        2) test_single_yolo_model "yolo11s" ;;
        3) test_single_yolo_model "yolo11m" ;;
        4) test_single_yolo_model "yolo11l" ;;
        5) test_single_yolo_model "yolo11x" ;;
        6) test_all_yolo_models ;;
        0) return ;;
        *) 
            print_error "잘못된 선택입니다."
            sleep 1
            test_yolo_models
            ;;
    esac
    
    echo ""
    read -p "계속하려면 Enter..."
}

# 단일 YOLO 모델 테스트
test_single_yolo_model() {
    local model_name="$1"
    print_info "$model_name 모델 테스트 중..."
    
    # 테스트 이미지 생성 (임시)
    python3 -c "
import numpy as np
from PIL import Image
import os

# 640x640 테스트 이미지 생성
test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
Image.fromarray(test_img).save('/tmp/test_image.jpg')
print('테스트 이미지 생성 완료')
"
    
    # YOLO 모델 테스트 실행
    python3 -c "
import time
import torch
from ultralytics import YOLO

try:
    print(f'🔄 {model_name} 모델 로딩 중...')
    start_time = time.time()
    model = YOLO('${model_name}.pt')
    load_time = time.time() - start_time
    print(f'✅ 모델 로딩 완료 ({load_time:.2f}초)')
    
    print(f'📊 모델 정보:')
    print(f'   - 디바이스: {model.device}')
    print(f'   - 모델 크기: {model_name}')
    
    print(f'🏃 추론 테스트 시작...')
    start_time = time.time()
    results = model('/tmp/test_image.jpg', verbose=False)
    inference_time = time.time() - start_time
    print(f'✅ 추론 완료 ({inference_time:.3f}초)')
    
    # 결과 분석
    if results:
        result = results[0]
        num_detections = len(result.boxes) if result.boxes is not None else 0
        print(f'🎯 검출 결과: {num_detections}개 객체 검출')
    
    print(f'✅ {model_name} 모델 테스트 성공!')
    
except Exception as e:
    print(f'❌ {model_name} 모델 테스트 실패: {str(e)}')
    exit(1)
" 2>/dev/null
    
    if [[ $? -eq 0 ]]; then
        print_success "$model_name 모델 테스트 완료!"
    else
        print_error "$model_name 모델 테스트 실패!"
    fi
    
    # 임시 파일 정리
    rm -f /tmp/test_image.jpg
}

# 전체 모델 테스트
test_all_yolo_models() {
    print_info "전체 YOLO 모델 테스트를 시작합니다..."
    echo ""
    
    for model in "${YOLO_MODEL_SIZES[@]}"; do
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        test_single_yolo_model "yolo11$model"
        echo ""
        sleep 1
    done
    
    print_success "전체 YOLO 모델 테스트 완료!"
}

# 성능 벤치마크 함수
run_performance_benchmark() {
    clear
    print_header "📊 성능 벤치마크"
    
    print_info "시스템 성능 벤치마크를 시작합니다..."
    
    # GPU 정보 수집
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo -e "${YELLOW}🎮 GPU 정보:${NC}"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits
        echo ""
    fi
    
    # PyTorch 벤치마크
    if python3 -c "import torch" &>/dev/null; then
        print_info "PyTorch 성능 벤치마크 실행 중..."
        
        python3 -c "
import torch
import time
import numpy as np

# 디바이스 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔥 PyTorch 벤치마크 (디바이스: {device})')

# 행렬 곱셈 벤치마크
sizes = [1000, 2000, 4000]
for size in sizes:
    print(f'\\n📊 행렬 크기: {size}x{size}')
    
    # CPU 벤치마크
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f'   CPU: {cpu_time:.3f}초')
    
    # GPU 벤치마크 (가능한 경우)
    if torch.cuda.is_available():
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # 워밍업
        torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f'   GPU: {gpu_time:.3f}초 (가속비: {speedup:.1f}x)')
    else:
        print(f'   GPU: 사용 불가')

print('\\n✅ PyTorch 벤치마크 완료!')
"
    fi
    
    # YOLO 추론 벤치마크
    if python3 -c "import ultralytics" &>/dev/null; then
        echo ""
        print_info "YOLO 추론 성능 벤치마크 실행 중..."
        
        python3 -c "
from ultralytics import YOLO
import time
import numpy as np
from PIL import Image

# 테스트 이미지 생성
test_images = []
for i in range(5):
    img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_images.append(Image.fromarray(img_array))

print(f'🎯 YOLO 추론 벤치마크')
models_to_test = ['yolo11n', 'yolo11s']

for model_name in models_to_test:
    try:
        print(f'\\n📦 {model_name} 모델:')
        model = YOLO(f'{model_name}.pt')
        
        # 워밍업
        model.predict(test_images[0], verbose=False)
        
        # 벤치마크
        start_time = time.time()
        for img in test_images:
            results = model.predict(img, verbose=False)
        total_time = time.time() - start_time
        
        fps = len(test_images) / total_time
        avg_time = total_time / len(test_images)
        
        print(f'   평균 추론 시간: {avg_time:.3f}초')
        print(f'   FPS: {fps:.1f}')
        
    except Exception as e:
        print(f'   ❌ {model_name} 테스트 실패: {e}')

print('\\n✅ YOLO 벤치마크 완료!')
"
    fi
    
    print_success "성능 벤치마크 완료!"
    echo ""
    read -p "계속하려면 Enter..."
}

# 문제 해결 도구
run_diagnostic_tools() {
    clear
    print_header "🔧 문제 해결 도구"
    
    echo "다음 도구 중 어떤 것을 실행하시겠습니까?"
    echo ""
    echo "1. 🔍 CUDA 연결 문제 진단"
    echo "2. 🧹 pip 캐시 및 임시 파일 정리"
    echo "3. 🔄 Python 패키지 충돌 해결"
    echo "4. 🎮 GPU 드라이버 상태 점검"
    echo "5. 📊 메모리 사용량 분석"
    echo "6. 🔧 전체 자동 진단 및 복구"
    echo "0. 메인 메뉴로 돌아가기"
    echo ""
    
    read -p "선택 (0-6): " tool_choice
    
    case $tool_choice in
        1) diagnose_cuda_issues ;;
        2) cleanup_system ;;
        3) resolve_package_conflicts ;;
        4) check_gpu_driver_status ;;
        5) analyze_memory_usage ;;
        6) auto_diagnose_and_fix ;;
        0) return ;;
        *)
            print_error "잘못된 선택입니다."
            sleep 1
            run_diagnostic_tools
            ;;
    esac
    
    echo ""
    read -p "계속하려면 Enter..."
    run_diagnostic_tools
}

# CUDA 연결 문제 진단
diagnose_cuda_issues() {
    print_info "CUDA 연결 문제를 진단합니다..."
    
    echo "1. NVIDIA 드라이버 상태:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,driver_version --format=csv
    else
        echo "   ❌ nvidia-smi 명령어 없음"
    fi
    
    echo ""
    echo "2. CUDA Toolkit 상태:"
    if command -v nvcc &>/dev/null; then
        nvcc --version | grep "release"
    else
        echo "   ❌ nvcc 명령어 없음"
    fi
    
    echo ""
    echo "3. PyTorch CUDA 인식:"
    python3 -c "
import torch
print(f'   PyTorch 버전: {torch.__version__}')
print(f'   CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA 버전: {torch.version.cuda}')
    print(f'   GPU 개수: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('   ❌ CUDA를 사용할 수 없습니다')
"
}

# 시스템 정리
cleanup_system() {
    print_info "시스템을 정리합니다..."
    
    # pip 캐시 정리
    echo "🧹 pip 캐시 정리 중..."
    pip cache purge
    
    # conda 캐시 정리 (설치되어 있는 경우)
    if command -v conda &>/dev/null; then
        echo "🧹 conda 캐시 정리 중..."
        conda clean -a -y
    fi
    
    # 임시 파일 정리
    echo "🧹 임시 파일 정리 중..."
    rm -rf /tmp/yolo_*
    rm -rf /tmp/test_image*
    
    # 로그 파일 정리 (오래된 것만)
    echo "🧹 오래된 로그 파일 정리 중..."
    find "$LOG_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    print_success "시스템 정리 완료!"
}

# 패키지 충돌 해결
resolve_package_conflicts() {
    print_info "Python 패키지 충돌을 해결합니다..."
    
    # 충돌 가능성이 높은 패키지 조합 확인
    python3 -c "
import pkg_resources
import sys

print('🔍 설치된 패키지 의존성 검사:')

# PyTorch 관련 패키지들
torch_packages = ['torch', 'torchvision', 'torchaudio']
for pkg in torch_packages:
    try:
        dist = pkg_resources.get_distribution(pkg)
        print(f'   {pkg}: {dist.version}')
    except:
        print(f'   {pkg}: 미설치')

# 버전 호환성 확인
try:
    import torch
    import torchvision
    print(f'\\n✅ PyTorch-torchvision 호환성: 정상')
except Exception as e:
    print(f'\\n❌ PyTorch-torchvision 호환성: {e}')

# CUDA 호환성 확인
try:
    import torch
    torch_cuda = torch.version.cuda
    if torch_cuda:
        print(f'✅ PyTorch CUDA 버전: {torch_cuda}')
    else:
        print(f'⚠️  PyTorch CPU 버전입니다')
except:
    print(f'❌ PyTorch CUDA 버전 확인 실패')
"
}

# GPU 드라이버 상태 점검
check_gpu_driver_status() {
    print_info "GPU 드라이버 상태를 점검합니다..."
    
    if command -v nvidia-smi &>/dev/null; then
        echo "📊 GPU 상태:"
        nvidia-smi
        echo ""
        
        echo "🔍 드라이버 모듈 확인:"
        lsmod | grep nvidia
        echo ""
        
        echo "🔍 NVIDIA 프로세스:"
        ps aux | grep nvidia
    else
        print_error "NVIDIA 드라이버가 설치되지 않았거나 로드되지 않았습니다."
    fi
}

# 메모리 사용량 분석
analyze_memory_usage() {
    print_info "메모리 사용량을 분석합니다..."
    
    echo "📊 시스템 메모리:"
    free -h
    echo ""
    
    if command -v nvidia-smi &>/dev/null; then
        echo "📊 GPU 메모리:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
        echo ""
    fi
    
    echo "📊 Python 프로세스 메모리 사용량:"
    ps aux | grep python | head -10
}

# 자동 진단 및 복구
auto_diagnose_and_fix() {
    print_info "자동 진단 및 복구를 시작합니다..."
    
    local issues_fixed=0
    
    # 1. pip 문제 해결
    if ! pip check &>/dev/null; then
        print_warning "pip 의존성 문제 발견. 해결 시도 중..."
        pip install --upgrade pip
        ((issues_fixed++))
    fi
    
    # 2. CUDA 경로 문제 해결
    if [[ ! -L "/usr/local/cuda" ]] && [[ -d "/usr/local/cuda-$(nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)" ]] 2>/dev/null; then
        print_warning "CUDA 심볼릭 링크 문제 발견. 해결 시도 중..."
        local cuda_version=$(nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)
        sudo ln -sf "/usr/local/cuda-$cuda_version" /usr/local/cuda
        ((issues_fixed++))
    fi
    
    # 3. 권한 문제 해결
    if [[ ! -w "$HOME/.cache" ]]; then
        print_warning "캐시 디렉토리 권한 문제 발견. 해결 시도 중..."
        chmod 755 "$HOME/.cache"
        ((issues_fixed++))
    fi
    
    if [[ $issues_fixed -eq 0 ]]; then
        print_success "자동으로 해결할 수 있는 문제가 발견되지 않았습니다."
    else
        print_success "$issues_fixed 개의 문제를 자동으로 해결했습니다."
    fi
}

# 버전 선택 함수
select_versions() {
    while true; do
        clear
        print_header "환경 버전 선택"
        echo -e "${CYAN}사용할 환경의 버전을 선택하세요${NC}"
        echo ""
        
        echo -e "${YELLOW}현재 선택된 버전:${NC}"
        echo "- NVIDIA Driver: ${REQUIRED_NVIDIA_DRIVER:-"미설정"}"
        echo "- CUDA Toolkit: ${REQUIRED_CUDA_TOOLKIT:-"미설정"}"
        echo "- PyTorch: ${SUPPORTED_VERSIONS[torch]:-"미설정"}"
        echo "- torchvision: ${SUPPORTED_VERSIONS[torchvision]:-"미설정"}"
        echo "- torchaudio: ${SUPPORTED_VERSIONS[torchaudio]:-"미설정"}"
        echo "- Ultralytics: ${SUPPORTED_VERSIONS[ultralytics]:-"미설정"}"
        echo "- numpy: ${SUPPORTED_VERSIONS[numpy]:-"미설정"}"
        echo "- opencv-python: ${SUPPORTED_VERSIONS[opencv-python]:-"미설정"}"
        echo "- pillow: ${SUPPORTED_VERSIONS[pillow]:-"미설정"}"
        echo ""
        
        echo "어떤 구성 요소의 버전을 변경하시겠습니까?"
        echo "1. NVIDIA 드라이버"
        echo "2. CUDA Toolkit"
        echo "3. PyTorch"
        echo "4. torchvision"
        echo "5. torchaudio"
        echo "6. Ultralytics"
        echo "7. numpy"
        echo "8. opencv-python"
        echo "9. pillow"
        echo "0. 메인 메뉴로 돌아가기"
        echo ""
        read -p "선택 (0-9): " component_choice
        
        case $component_choice in
            1)
                select_nvidia_driver
                ;;
            2)
                select_cuda_toolkit
                ;;
            3)
                select_pytorch
                ;;
            4)
                select_torchvision
                ;;
            5)
                select_torchaudio
                ;;
            6)
                select_ultralytics
                ;;
            7)
                select_numpy
                ;;
            8)
                select_opencv
                ;;
            9)
                select_pillow
                ;;
            0)
                break
                ;;
            *)
                print_error "잘못된 선택입니다."
                sleep 1
                ;;
        esac
        
        if [[ $component_choice != 0 ]]; then
            echo ""
            read -p "다른 구성 요소도 변경하시겠습니까? (y/N): " continue_choice
            if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
                break
            fi
        fi
    done
    
    CURRENT_VERSIONS_SET=1
}

# NVIDIA 드라이버 선택
select_nvidia_driver() {
    echo ""
    echo -e "${YELLOW}NVIDIA 드라이버 선택:${NC}"
    for key in $(echo "${!NVIDIA_DRIVER_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${NVIDIA_DRIVER_OPTIONS[$key]}"
        echo "$key. nvidia-driver-$version - $desc"
    done
    echo ""
    read -p "NVIDIA 드라이버 선택 (1-16): " nvidia_choice
    
    if [[ -n "${NVIDIA_DRIVER_OPTIONS[$nvidia_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${NVIDIA_DRIVER_OPTIONS[$nvidia_choice]}"
        REQUIRED_NVIDIA_DRIVER="$version"
        print_success "NVIDIA 드라이버: nvidia-driver-$version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# CUDA Toolkit 선택
select_cuda_toolkit() {
    echo ""
    echo -e "${YELLOW}CUDA Toolkit 선택:${NC}"
    for key in $(echo "${!CUDA_TOOLKIT_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${CUDA_TOOLKIT_OPTIONS[$key]}"
        echo "$key. CUDA $version - $desc"
    done
    echo ""
    read -p "CUDA Toolkit 선택 (1-12): " cuda_choice
    
    if [[ -n "${CUDA_TOOLKIT_OPTIONS[$cuda_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${CUDA_TOOLKIT_OPTIONS[$cuda_choice]}"
        REQUIRED_CUDA_TOOLKIT="$version"
        print_success "CUDA Toolkit: $version 선택됨"
        
        # 다중 CUDA 관리 환경 구성 제안
        echo ""
        echo -e "${CYAN}다중 CUDA 버전 관리를 위한 환경 구성을 권장합니다.${NC}"
        read -p "CUDA alternatives 설정 및 다중 버전 관리 환경을 구성하시겠습니까? (y/n): " setup_multi
        
        if [[ "$setup_multi" == "y" || "$setup_multi" == "Y" ]]; then
            echo -e "${GREEN}다중 CUDA 관리 환경을 구성합니다...${NC}"
            setup_cuda_alternatives
            echo ""
            echo -e "${YELLOW}추가 CUDA 버전 설치를 원하시면 '2. CUDA 버전 관리' 메뉴를 이용하세요.${NC}"
        fi
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# PyTorch 선택
select_pytorch() {
    echo ""
    echo -e "${YELLOW}PyTorch 선택:${NC}"
    for key in $(echo "${!PYTORCH_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${PYTORCH_OPTIONS[$key]}"
        echo "$key. PyTorch $version - $desc"
    done
    echo ""
    read -p "PyTorch 선택 (1-5): " pytorch_choice
    
    if [[ -n "${PYTORCH_OPTIONS[$pytorch_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${PYTORCH_OPTIONS[$pytorch_choice]}"
        SUPPORTED_VERSIONS["torch"]="$version"
        print_success "PyTorch: $version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# torchvision 선택
select_torchvision() {
    echo ""
    echo -e "${YELLOW}torchvision 선택:${NC}"
    for key in $(echo "${!TORCHVISION_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${TORCHVISION_OPTIONS[$key]}"
        echo "$key. torchvision $version - $desc"
    done
    echo ""
    read -p "torchvision 선택 (1-5): " torchvision_choice
    
    if [[ -n "${TORCHVISION_OPTIONS[$torchvision_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${TORCHVISION_OPTIONS[$torchvision_choice]}"
        SUPPORTED_VERSIONS["torchvision"]="$version"
        print_success "torchvision: $version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# torchaudio 선택
select_torchaudio() {
    echo ""
    echo -e "${YELLOW}torchaudio 선택:${NC}"
    for key in $(echo "${!TORCHAUDIO_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${TORCHAUDIO_OPTIONS[$key]}"
        echo "$key. torchaudio $version - $desc"
    done
    echo ""
    read -p "torchaudio 선택 (1-5): " torchaudio_choice
    
    if [[ -n "${TORCHAUDIO_OPTIONS[$torchaudio_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${TORCHAUDIO_OPTIONS[$torchaudio_choice]}"
        SUPPORTED_VERSIONS["torchaudio"]="$version"
        print_success "torchaudio: $version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# Ultralytics 선택
select_ultralytics() {
    echo ""
    echo -e "${YELLOW}Ultralytics 선택:${NC}"
    for key in $(echo "${!ULTRALYTICS_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${ULTRALYTICS_OPTIONS[$key]}"
        echo "$key. ultralytics $version - $desc"
    done
    echo ""
    read -p "Ultralytics 선택 (1-3): " ultralytics_choice
    
    if [[ -n "${ULTRALYTICS_OPTIONS[$ultralytics_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${ULTRALYTICS_OPTIONS[$ultralytics_choice]}"
        SUPPORTED_VERSIONS["ultralytics"]="$version"
        print_success "ultralytics: $version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# numpy 선택
select_numpy() {
    echo ""
    echo -e "${YELLOW}numpy 선택:${NC}"
    for key in $(echo "${!NUMPY_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${NUMPY_OPTIONS[$key]}"
        echo "$key. numpy $version - $desc"
    done
    echo ""
    read -p "numpy 선택 (1-3): " numpy_choice
    
    if [[ -n "${NUMPY_OPTIONS[$numpy_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${NUMPY_OPTIONS[$numpy_choice]}"
        SUPPORTED_VERSIONS["numpy"]="$version"
        print_success "numpy: $version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# opencv-python 선택
select_opencv() {
    echo ""
    echo -e "${YELLOW}opencv-python 선택:${NC}"
    for key in $(echo "${!OPENCV_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${OPENCV_OPTIONS[$key]}"
        echo "$key. opencv-python $version - $desc"
    done
    echo ""
    read -p "opencv-python 선택 (1-3): " opencv_choice
    
    if [[ -n "${OPENCV_OPTIONS[$opencv_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${OPENCV_OPTIONS[$opencv_choice]}"
        SUPPORTED_VERSIONS["opencv-python"]="$version"
        print_success "opencv-python: $version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# pillow 선택
select_pillow() {
    echo ""
    echo -e "${YELLOW}pillow 선택:${NC}"
    for key in $(echo "${!PILLOW_OPTIONS[@]}" | tr ' ' '\n' | sort -n); do
        IFS='|' read -r version desc <<< "${PILLOW_OPTIONS[$key]}"
        echo "$key. pillow $version - $desc"
    done
    echo ""
    read -p "pillow 선택 (1-3): " pillow_choice
    
    if [[ -n "${PILLOW_OPTIONS[$pillow_choice]}" ]]; then
        IFS='|' read -r version desc <<< "${PILLOW_OPTIONS[$pillow_choice]}"
        SUPPORTED_VERSIONS["pillow"]="$version"
        print_success "pillow: $version 선택됨"
    else
        print_error "잘못된 선택입니다. 현재 설정을 유지합니다."
    fi
}

# 개별 패키지 선택 설치 함수
selective_install() {
    print_header "개별 패키지 선택 설치"
    
    while true; do
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}설치할 구성 요소를 선택하세요${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        echo ""
        echo -e "${YELLOW}시스템 구성 요소:${NC}"
        echo "1. NVIDIA 드라이버 (${REQUIRED_NVIDIA_DRIVER:-"버전 미설정"})"
        echo "2. CUDA Toolkit (${REQUIRED_CUDA_TOOLKIT:-"버전 미설정"})"
        
        echo ""
        echo -e "${YELLOW}Python 패키지:${NC}"
        echo "3. torch (${SUPPORTED_VERSIONS[torch]:-"버전 미설정"})"
        echo "4. torchvision (${SUPPORTED_VERSIONS[torchvision]:-"버전 미설정"})"
        echo "5. torchaudio (${SUPPORTED_VERSIONS[torchaudio]:-"버전 미설정"})"
        echo "6. ultralytics (${SUPPORTED_VERSIONS[ultralytics]:-"버전 미설정"})"
        echo "7. numpy (${SUPPORTED_VERSIONS[numpy]:-"버전 미설정"})"
        echo "8. opencv-python (${SUPPORTED_VERSIONS[opencv-python]:-"버전 미설정"})"
        echo "9. pillow (${SUPPORTED_VERSIONS[pillow]:-"버전 미설정"})"
        
        echo ""
        echo "10. pip 캐시 정리"
        echo "0. 메인 메뉴로 돌아가기"
        
        echo ""
        read -p "선택 (0-10): " install_choice
        
        case $install_choice in
            1)
                install_nvidia_driver
                ;;
            2)
                install_cuda_toolkit
                ;;
            3)
                install_single_package "torch"
                ;;
            4)
                install_single_package "torchvision"
                ;;
            5)
                install_single_package "torchaudio"
                ;;
            6)
                install_single_package "ultralytics"
                ;;
            7)
                install_single_package "numpy"
                ;;
            8)
                install_single_package "opencv-python"
                ;;
            9)
                install_single_package "pillow"
                ;;
            10)
                clean_pip_cache
                ;;
            0)
                break
                ;;
            *)
                print_error "잘못된 선택입니다."
                sleep 1
                ;;
        esac
        
        echo ""
        read -p "계속하려면 Enter..."
    done
}

# NVIDIA 드라이버 설치 함수
install_nvidia_driver() {
    if [[ -z "$REQUIRED_NVIDIA_DRIVER" ]]; then
        print_warning "NVIDIA 드라이버 버전이 설정되지 않았습니다."
        echo -e "${CYAN}먼저 '3. 패키지 버전 구성' 메뉴에서 버전을 선택해주세요.${NC}"
        return 1
    fi
    
    print_info "NVIDIA 드라이버 설치를 시작합니다..."
    local driver_version="$REQUIRED_NVIDIA_DRIVER"
    local driver_package="nvidia-driver-$driver_version"
    
    print_info "설치할 패키지: $driver_package"
    sudo apt update && sudo apt install -y "$driver_package"
    local exit_code=$?
    
    if handle_command_result $exit_code "NVIDIA 드라이버 설치" "$driver_package"; then
        print_info "시스템 재부팅 후 적용됩니다."
    fi
}

# CUDA Toolkit 설치 함수
install_cuda_toolkit() {
    if [[ -z "$REQUIRED_CUDA_TOOLKIT" ]]; then
        print_warning "CUDA Toolkit 버전이 설정되지 않았습니다."
        echo -e "${CYAN}먼저 '3. 패키지 버전 구성' 메뉴에서 버전을 선택해주세요.${NC}"
        return 1
    fi
    
    print_info "CUDA Toolkit 설치를 시작합니다..."
    local cuda_version="$REQUIRED_CUDA_TOOLKIT"
    local cuda_package
    
    # CUDA 버전에 따라 패키지명 결정
    case "$cuda_version" in
        "13.0")
            cuda_package="cuda-toolkit-13-0"
            ;;
        "12.9")
            cuda_package="cuda-toolkit-12-9"
            ;;
        "12.8")
            cuda_package="cuda-toolkit-12-8"
            ;;
        "12.6")
            cuda_package="cuda-toolkit-12-6"
            ;;
        "12.5")
            cuda_package="cuda-toolkit-12-5"
            ;;
        "12.4")
            cuda_package="cuda-toolkit-12-4"
            ;;
        "12.3")
            cuda_package="cuda-toolkit-12-3"
            ;;
        "12.2")
            cuda_package="cuda-toolkit-12-2"
            ;;
        "12.1")
            cuda_package="cuda-toolkit-12-1"
            ;;
        "12.0")
            cuda_package="cuda-toolkit-12-0"
            ;;
        "11.8")
            cuda_package="cuda-toolkit-11-8"
            ;;
        "11.7")
            cuda_package="cuda-toolkit-11-7"
            ;;
        *)
            cuda_package="cuda-toolkit-${cuda_version//./-}"
            ;;
    esac
    
    # NVIDIA CUDA 저장소가 있는지 확인
    if ! grep -q "developer.download.nvidia.com" /etc/apt/sources.list.d/* 2>/dev/null; then
        print_info "NVIDIA CUDA 저장소를 추가합니다..."
        
        # CUDA 키링 다운로드 및 설치
        if [[ ! -f "cuda-keyring_1.0-1_all.deb" ]]; then
            wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        fi
        
        sudo dpkg -i cuda-keyring_1.0-1_all.deb >/dev/null 2>&1
        
        # 오래된 CUDA GPG 키 제거 (경고 메시지 방지)
        sudo apt-key del 7fa2af80 >/dev/null 2>&1
        
        print_info "패키지 목록을 업데이트합니다..."
        sudo apt update >/dev/null 2>&1
    fi
    
    print_info "설치할 패키지: $cuda_package"
    sudo apt install -y "$cuda_package"
    
    if [[ $? -eq 0 ]]; then
        print_success "CUDA Toolkit 설치 완료!"
        # 설치 후 키링 파일 정리
        rm -f cuda-keyring_1.0-1_all.deb
    else
        print_error "CUDA Toolkit 설치 실패!"
    fi
}

# 특정 버전 CUDA Toolkit 설치 함수
install_cuda_toolkit_version() {
    local version="$1"
    print_info "CUDA Toolkit $version 설치를 시작합니다..."
    
    # NVIDIA 저장소 설정 확인
    if ! grep -q "developer.download.nvidia.com" /etc/apt/sources.list.d/* 2>/dev/null; then
        echo -e "${CYAN}NVIDIA 저장소 설정 중...${NC}"
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt update
    fi
    
    # 특정 버전 설치
    echo -e "${CYAN}CUDA Toolkit $version 설치 중...${NC}"
    sudo apt install -y "cuda-toolkit-${version//./-}"
    
    if [[ $? -eq 0 ]]; then
        print_success "CUDA Toolkit $version 설치 완료!"
    else
        print_error "CUDA Toolkit $version 설치 실패!"
    fi
}

# 단일 패키지 설치 함수
install_single_package() {
    local package="$1"
    
    # 버전이 설정되지 않은 경우 사용자에게 안내
    if [[ -z "${SUPPORTED_VERSIONS[$package]}" ]]; then
        print_warning "$package의 버전이 설정되지 않았습니다."
        echo -e "${CYAN}먼저 '3. 패키지 버전 구성' 메뉴에서 버전을 선택하거나,${NC}"
        echo -e "${CYAN}최신 버전을 설치하시겠습니까?${NC}"
        echo ""
        read -p "최신 버전 설치 (y/n): " install_latest
        
        if [[ "$install_latest" != "y" && "$install_latest" != "Y" ]]; then
            print_info "설치가 취소되었습니다."
            return 0
        fi
        
        # 최신 버전 설치
        print_info "$package 최신 버전 설치 중..."
        
        # PyTorch 관련 패키지는 특별한 설치 방법 사용
        if [[ "$package" == "torch" || "$package" == "torchvision" || "$package" == "torchaudio" ]]; then
            # 기존 PyTorch 패키지들 제거
            pip uninstall -y torch torchvision torchaudio
            
            # PyTorch 공식 저장소에서 최신 CUDA 버전 설치
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            # 일반 패키지 최신 버전 설치
            pip uninstall -y "$package"
            pip install "$package"
        fi
        
        if [[ $? -eq 0 ]]; then
            print_success "$package 최신 버전 설치 완료!"
        else
            print_error "$package 설치 실패!"
        fi
        return
    fi
    
    local version="${SUPPORTED_VERSIONS[$package]}"
    
    print_info "$package==$version 설치 중..."
    
    # PyTorch 관련 패키지는 특별한 설치 방법 사용
    if [[ "$package" == "torch" || "$package" == "torchvision" || "$package" == "torchaudio" ]]; then
        # 기존 PyTorch 패키지들 제거
        pip uninstall -y torch torchvision torchaudio
        
        # PyTorch 공식 저장소에서 설치
        if [[ "$version" == *"+cu124"* ]]; then
            # CUDA 12.4 버전
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        elif [[ "$version" == *"+cu121"* ]]; then
            # CUDA 12.1 버전
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$version" == *"+cu118"* ]]; then
            # CUDA 11.8 버전
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            # CPU 버전 또는 기본 버전
            pip install torch torchvision torchaudio
        fi
    else
        # 일반 패키지 설치
        pip uninstall -y "$package"
        pip install "$package==$version"
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "$package==$version 설치 완료!"
    else
        print_error "$package==$version 설치 실패!"
    fi
}

# pip 캐시 정리 함수
clean_pip_cache() {
    print_info "pip 캐시를 정리합니다..."
    pip cache purge
    print_success "pip 캐시 정리 완료!"
}

# ============================================================================
# 메인 스크립트 실행부
# ============================================================================

# 초기화 함수
initialize_system() {
    log_message "INFO" "YOLO 환경 관리 시스템 시작 - 세션 ID: $SESSION_ID"
    
    # 시스템 요구사항 확인
    check_system_requirements
    
    # 초기 환경 정보 로깅
    log_message "INFO" "시스템 정보 - OS: $(lsb_release -d | cut -f2), Memory: ${SYSTEM_MEMORY_GB}GB, GPU: ${GPU_MEMORY_GB}GB"
    
    # 환경 백업
    backup_system_state
    
    print_info "🚀 YOLO/딥러닝 환경 관리 시스템이 시작되었습니다."
    echo -e "${CYAN}📝 세션 ID: $SESSION_ID${NC}"
    echo -e "${CYAN}📄 로그 파일: $LOG_FILE${NC}"
    echo ""
    sleep 1
}

# 종료 정리 함수
cleanup_on_exit() {
    local exit_code=$?
    
    log_message "INFO" "시스템 종료 - 세션 ID: $SESSION_ID, 종료 코드: $exit_code"
    
    # 종료 시 시스템 상태 저장
    if [[ $exit_code -eq 0 ]]; then
        backup_system_state
    fi
    
    echo -e "${CYAN}🎯 YOLO 환경 관리 시스템을 안전하게 종료했습니다.${NC}"
    echo -e "${CYAN}📝 로그 및 백업 파일은 다음 위치에 저장되었습니다:${NC}"
    echo -e "${CYAN}   📄 로그: $LOG_FILE${NC}"
    echo -e "${CYAN}   📂 백업: $BACKUP_DIR${NC}"
    
    exit $exit_code
}

# 멀티코어 최적화 설정
optimize_multicore() {
    print_header "🚀 멀티코어 최적화 설정"
    
    local cpu_cores=$(nproc)
    print_info "감지된 CPU 코어 수: $cpu_cores"
    
    # 환경변수 설정
    export OMP_NUM_THREADS=$cpu_cores
    export MKL_NUM_THREADS=$cpu_cores
    export NUMEXPR_NUM_THREADS=$cpu_cores
    export OPENBLAS_NUM_THREADS=$cpu_cores
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Conda 환경에 영구 설정
    if [[ -n "$CONDA_PREFIX" ]]; then
        local env_vars_file="$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
        mkdir -p "$(dirname "$env_vars_file")"
        
        cat > "$env_vars_file" << EOF
#!/bin/bash
# 멀티코어 최적화 환경변수
export OMP_NUM_THREADS=$cpu_cores
export MKL_NUM_THREADS=$cpu_cores
export NUMEXPR_NUM_THREADS=$cpu_cores
export OPENBLAS_NUM_THREADS=$cpu_cores
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF
        chmod +x "$env_vars_file"
        print_success "Conda 환경에 멀티코어 설정 영구 저장: $env_vars_file"
    fi
    
    # PyTorch 설정 확인
    print_info "PyTorch 멀티코어 설정 확인 중..."
    
    if command -v python3 &>/dev/null; then
        python3 -c "
import torch
import multiprocessing as mp
print('CPU 코어 수: {}'.format(mp.cpu_count()))
print('PyTorch CPU 스레드: {}'.format(torch.get_num_threads()))
print('OpenMP 스레드: {}'.format(torch.get_num_interop_threads()))
if torch.cuda.is_available():
    print('CUDA 장치 수: {}'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        print('GPU {}: {}'.format(i, torch.cuda.get_device_name(i)))
" 2>/dev/null || {
            print_warning "PyTorch 설정 확인을 건너뜁니다."
        }
    else
        print_warning "Python3를 찾을 수 없습니다."
    fi
    
    print_success "멀티코어 최적화 설정 완료"
    echo ""
    print_info "현재 설정된 환경변수:"
    echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS:-없음}"
    echo "  MKL_NUM_THREADS: ${MKL_NUM_THREADS:-없음}"
    echo "  NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS:-없음}"
    echo "  OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS:-없음}"
    echo ""
}

# YOLO 학습 최적화 실행
run_optimized_yolo() {
    local script_path="$1"
    
    if [[ ! -f "$script_path" ]]; then
        print_error "스크립트를 찾을 수 없습니다: $script_path"
        return 1
    fi
    
    print_header "🎯 최적화된 YOLO 학습 실행"
    
    # 멀티코어 최적화 적용
    optimize_multicore
    
    # CPU 친화성 설정으로 모든 코어 사용
    print_info "모든 CPU 코어를 사용하여 YOLO 학습 시작..."
    taskset -c 0-$(($(nproc)-1)) python3 "$script_path"
}

# 신호 처리기 설정
trap cleanup_on_exit EXIT
trap 'echo ""; print_warning "중단 신호 감지됨. 안전하게 종료 중..."; cleanup_on_exit' INT TERM

# 메인 실행
main() {
    # 시스템 초기화
    initialize_system
    
    # 메인 메뉴 실행
    show_menu
}

# 스크립트 직접 실행 시에만 main 함수 호출
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

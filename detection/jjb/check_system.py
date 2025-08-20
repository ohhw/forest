import psutil
import torch
import subprocess


def check_system_resources():
    """시스템 리소스 상태 확인"""
    
    print("=== 시스템 리소스 상태 ===")
    
    # CPU 정보
    print(f"CPU 사용률: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"CPU 코어 수: {psutil.cpu_count()}")
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"총 메모리: {memory.total / (1024**3):.1f}GB")
    print(f"사용 가능한 메모리: {memory.available / (1024**3):.1f}GB")
    print(f"메모리 사용률: {memory.percent:.1f}%")
    
    # GPU 정보
    if torch.cuda.is_available():
        print(f"\nGPU 사용 가능: Yes")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  총 메모리: {props.total_memory / (1024**3):.1f}GB")
            
            # 현재 GPU 메모리 사용량
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  사용 중인 메모리: {memory_allocated:.1f}GB")
                print(f"  예약된 메모리: {memory_reserved:.1f}GB")
    else:
        print(f"\nGPU 사용 가능: No")
    
    # 디스크 공간 확인
    disk_usage = psutil.disk_usage('/')
    print(f"\n디스크 사용량:")
    print(f"  총 용량: {disk_usage.total / (1024**3):.1f}GB")
    print(f"  사용 중: {disk_usage.used / (1024**3):.1f}GB")
    print(f"  사용 가능: {disk_usage.free / (1024**3):.1f}GB")
    print(f"  사용률: {(disk_usage.used / disk_usage.total) * 100:.1f}%")


def get_recommended_settings():
    """권장 설정값 제안"""
    
    print("\n=== 권장 설정값 ===")
    
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # 배치 크기 권장값
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory >= 24:
            recommended_batch = 32
        elif gpu_memory >= 16:
            recommended_batch = 16
        elif gpu_memory >= 12:
            recommended_batch = 8
        elif gpu_memory >= 8:
            recommended_batch = 4
        else:
            recommended_batch = 2
    else:
        recommended_batch = 2
    
    # 워커 수 권장값
    cpu_cores = psutil.cpu_count()
    recommended_workers = min(cpu_cores // 2, 8)  # CPU 코어의 절반, 최대 8
    
    print(f"권장 배치 크기: {recommended_batch}")
    print(f"권장 워커 수: {recommended_workers}")
    
    # 메모리 부족 시 추가 권장사항
    if available_gb < 8:
        print("\n⚠️  메모리 부족 감지:")
        print("  - cache=False 설정 권장")
        print("  - 배치 크기를 더 줄여보세요")
        print("  - 다른 프로그램을 종료해보세요")


if __name__ == "__main__":
    check_system_resources()
    get_recommended_settings()

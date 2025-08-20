import os
import torch
import torch.hub
from ultralytics import YOLO
import gc
import warnings

# 선택적 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 메모리 안전 환경변수 설정 (expandable_segments 제거)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'  # expandable_segments 제거
os.environ['TORCH_SERIALIZATION_WEIGHTS_ONLY'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'  # 스레드 수 더 감소
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 동기화 강제

# 멀티프로세싱 최적화
torch.multiprocessing.set_sharing_strategy('file_system')

# GPU 설정 (안정성 우선)
torch.backends.cudnn.enabled = False  # True → False (안정성 우선)
torch.backends.cudnn.benchmark = False  # True → False
torch.backends.cudnn.deterministic = True  # False → True
torch.backends.cuda.matmul.allow_tf32 = False  # True → False
torch.backends.cudnn.allow_tf32 = False  # True → False

# 초기 설정
Date = "250805"
Time = "13h30m"  # 시간 업데이트
User = "hwoh"
Product = "jjb"
yolo_nm = "11l"
model_nm = f"test_{Product}_dod_{yolo_nm}_{Date}{Time}_ultra_safe"  # 울트라 안전 모드
data_nm = "v10"

os.chdir(f"/home/{User}/detection/{Product}")
torch.hub.set_dir(f"/home/{User}/detection/{Product}")

print("🚨🚨🚨 울트라 안전 모드 - PyTorch 내부 오류 방지 🚨🚨🚨")
print("⚙️  설정: expandable_segments 제거, cuDNN 비활성화, 최소 배치")

# 강력한 메모리 정리 함수
def aggressive_memory_cleanup():
    if torch.cuda.is_available():
        for i in range(5):  # 3 → 5번으로 증가
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect()
    
    # 시스템 메모리 정리 시도
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

# 메모리 정리 실행
aggressive_memory_cleanup()

# YOLO 가중치 경로 설정
try:
    model = YOLO(f"/home/{User}/detection/{Product}/yolo{yolo_nm}.pt")
    print(f"로컬 모델 로딩 성공: yolo{yolo_nm}.pt")
except Exception as e:
    print(f"로컬 모델 로딩 실패: {e}")
    print("Ultralytics Hub에서 모델 다운로드 중...")
    model = YOLO(f"yolo{yolo_nm}.pt")

# 모델 설정
DO_TRAIN = True
DO_VAL = False
DO_PREDICT = False

if DO_TRAIN:
    # GPU 상태 확인
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        aggressive_memory_cleanup()  # 추가 메모리 정리
        
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory:.2f}GB")
        print(f"Available: {total_memory - cached:.2f}GB")
        
        # RTX 4090 특화 울트라 안전 설정
        if total_memory > 40:  # A100 등 고성능 GPU
            batch_size = 16
            workers = 0   # 2 → 0으로 변경 (segfault 방지)
            cache = False
            print("🚀 고성능 GPU 감지 - 울트라 안전 설정")
        elif total_memory > 20:  # RTX 3090/4090 등
            batch_size = 10
            workers = 0   # 1 → 0으로 변경 (segfault 완전 방지)
            cache = False
            print("🚀 RTX 4090 감지 - 최대 안전 설정")
        else:  # 일반 GPU
            batch_size = 6
            workers = 0   # 1 → 0으로 변경
            cache = False
            print("🚀 일반 GPU 감지 - 최소 설정")
            
    else:
        print("CUDA not available, using CPU")
        batch_size = 4
        workers = 0  # 1 → 0으로 변경
        cache = False
    
    print(f"🚨 울트라 안전 모드 학습 시작 - batch={batch_size}, workers={workers}")
    
    # 울트라 안전 학습 설정
    model.train(
        data=f"/hdd/datasets/dod_data/{Product}/{Product}_defect_detection_data_{data_nm}.yaml",
        epochs=300,
        batch=batch_size,  # 매우 안전한 배치 크기
        patience=100,
        dropout=0.22,
        iou=0.44, 
        lr0=0.0005,  # 0.001 → 0.0005로 감소
        lrf=0.00001,  # 0.0001 → 0.00001로 감소
        optimizer="AdamW",
        
        # 🚨 울트라 안전 설정
        workers=workers,  # 최소 워커
        cache=cache,      # 캐시 완전 비활성화
        rect=False,       # True → False (rectangular training 비활성화)
        amp=False,        # True → False (Mixed Precision 비활성화)
        half=False,       # FP16 비활성화
        
        # 모든 데이터 증강 최소화
        close_mosaic=10,  # 50 → 10으로 더 빨리 비활성화
        mixup=0.0,        # 0.05 → 0.0 완전 비활성화
        copy_paste=0.0,   # 0.05 → 0.0 완전 비활성화
        degrees=0.0,      # 10.0 → 0.0 완전 비활성화
        translate=0.0,    # 0.1 → 0.0 완전 비활성화
        scale=0.1,        # 0.3 → 0.1 최소화
        shear=0.0,        # 2.0 → 0.0 완전 비활성화
        perspective=0.0,  # 0.0001 → 0.0 완전 비활성화
        flipud=0.0,
        fliplr=0.2,       # 0.5 → 0.2로 감소
        mosaic=0.0,       # 1.0 → 0.0 완전 비활성화
        hsv_h=0.0,        # 0.015 → 0.0 완전 비활성화
        hsv_s=0.0,        # 0.7 → 0.0 완전 비활성화
        hsv_v=0.0,        # 0.4 → 0.0 완전 비활성화
        auto_augment=None, # "randaugment" → None
        erasing=0.0,
        crop_fraction=1.0,
        
        # 추가 안전 설정
        single_cls=False,
        multi_scale=False,
        overlap_mask=True,
        mask_ratio=4,
        val=True,
        plots=False,      # True → False (메모리 절약)
        save_period=50,   # 25 → 50으로 증가
        
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect",
        name=f"{model_nm}",
        verbose=True,
    )

    # 학습 완료 후 메모리 정리
    aggressive_memory_cleanup()

best_weight_path = (
    f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/best.pt"
)

print("🎉 울트라 안전 모드 학습 완료!")
print(f"Best 가중치 저장 경로: {best_weight_path}")

# 나머지 코드는 동일...
if DO_VAL or DO_PREDICT:
    if not os.path.exists(best_weight_path):
        print(f"[ERROR] best.pt 파일이 존재하지 않습니다: {best_weight_path}")
        exit(1)
    model_Product = YOLO(best_weight_path)

if DO_VAL:
    metrics = model_Product.val()
    print(metrics)

# 예측 설정
pre_cls = 1.0
pre_kobj = 1.0
pred_conf = 0.5
pred_dfl = 1

if DO_PREDICT:
    # 예측 수행
    results = model_Product.predict(
        f"/hdd/datasets/dod_data/{Product}/{data_nm}/val/images",
        save=True,
        save_crop=True,
        save_txt=True,
        cls=pre_cls,
        kobj=pre_kobj,
        conf=pred_conf,
        dfl=pred_dfl,
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
        name=f"pred_{model_nm}_val",
    )

    # 신뢰도 없이 예측
    results = model_Product.predict(
        f"/hdd/datasets/dod_data/{Product}/{data_nm}/val/images",
        save=True,
        save_crop=True,
        save_txt=True,
        show_conf=False,
        cls=pre_cls,
        kobj=pre_kobj,
        conf=pred_conf,
        dfl=pred_dfl,
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
        name=f"pred_{model_nm}_val_without_conf",
    )
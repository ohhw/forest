import os
import torch
import torch.hub
from ultralytics import YOLO
import gc

# # 강화된 메모리 관리 및 안정성 설정
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # 더 작게
# os.environ['TORCH_SERIALIZATION_WEIGHTS_ONLY'] = '0'  # weights_only 문제 해결
# os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP 쓰레드 제한
# os.environ['MKL_NUM_THREADS'] = '4'  # MKL 쓰레드 제한

# # GPU 메모리 및 OpenCV 최적화 설정 (더 보수적으로)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cuda.matmul.allow_tf32 = False  # 정확성 우선
# torch.backends.cudnn.allow_tf32 = False

# # 멀티프로세싱 안전 설정
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn', force=True)  # 추가

# 초기 설정
Date = "250807"
Time = "10h30m"  # 시간 업데이트
User = "hwoh"
Product = "csn"
yolo_nm = "11n"
# model_nm = f"test_{Product}_dod_{yolo_nm}_{Date}{Time}_safe"  # 안전 모드 표시
model_nm = f"{Product}_dod_{yolo_nm}_{Date}{Time}"
data_nm = "v5"

os.chdir(f"/home/{User}/detection/{Product}")
torch.hub.set_dir(f"/home/{User}/detection/{Product}")

# YOLO 가중치 경로 설정 - 공통 폴더 사용
try:
    # 공통 가중치 폴더에서 먼저 찾기
    common_model_path = f"/home/{User}/yolo_models/yolo{yolo_nm}.pt"
    if os.path.exists(common_model_path):
        model = YOLO(common_model_path)
        print(f"✅ 공통 모델 로딩 성공: {common_model_path}")
    else:
        # 공통 폴더에 없으면 로컬에서 찾기
        local_model_path = f"/home/{User}/detection/{Product}/yolo{yolo_nm}.pt"
        if os.path.exists(local_model_path):
            model = YOLO(local_model_path)
            print(f"✅ 로컬 모델 로딩 성공: {local_model_path}")
        else:
            raise FileNotFoundError("로컬 모델 파일이 존재하지 않음")
            
except Exception as e:
    print(f"❌ 로컬 모델 로딩 실패: {e}")
    print("🌐 Ultralytics Hub에서 모델 다운로드 중...")
    model = YOLO(f"yolo{yolo_nm}.pt")
    
    # 다운로드 성공하면 공통 폴더에 저장
    try:
        os.makedirs("/home/hwoh/yolo_models", exist_ok=True)
        model.save(f"/home/{User}/yolo_models/yolo{yolo_nm}.pt")
        print(f"📦 모델을 공통 폴더에 저장: /home/{User}/yolo_models/yolo{yolo_nm}.pt")
    except Exception as save_error:
        print(f"⚠️ 공통 폴더 저장 실패: {save_error}")
        
# 모델 설정
DO_TRAIN = True
DO_VAL = False
DO_PREDICT = False  # 학습 완료 후 활성화

if DO_TRAIN:
    # 강력한 GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        for i in range(3):  # 3번 반복 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        # GPU 상태 확인
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"GPU: {gpu_name}")
        print(f"Device: {device}")
        print(f"Total Memory: {total_memory:.2f}GB")
        print(f"Allocated: {allocated:.2f}GB")
        print(f"Cached: {cached:.2f}GB")
        print(f"Available: {total_memory - cached:.2f}GB")
    else:
        print("CUDA not available, using CPU")
    
    # print("🚨 Segmentation fault 방지 모드로 학습 시작...")
    # print("⚙️  안전 설정: workers=0, batch=16, 모든 증강 최소화")
    
    # 안전 모드 학습 (Segmentation fault 방지)
    model.train(
        data=f"/hdd/datasets/dod_data/{Product}/{Product}_defect_detection_data_{data_nm}.yaml",
        epochs=300,
        batch=16,  # 더 안전한 배치 크기
        patience=100,
        dropout=0.22,
        iou=0.44, 
        lr0=0.0005,
        lrf=0.00001,
        optimizer="AdamW",
        
        # # 🚨 Segmentation fault 방지 핵심 설정
        # workers=0,  # 멀티프로세싱 완전 비활성화
        # cache=False,  # 캐시 비활성화
        # rect=False,  # rectangular training 비활성화
        # amp=False,  # AMP 비활성화 (안정성 우선)
        
        # # 데이터 증강 최소화 (메모리 접근 오류 방지)
        # close_mosaic=10,
        # mixup=0.0,
        # copy_paste=0.0,
        # degrees=0.0,  # 회전 완전 비활성화
        # translate=0.0,  # 이동 완전 비활성화
        # scale=0.1,  # 스케일 최소화
        # shear=0.0,  # 전단 변환 비활성화
        # perspective=0.0,  # perspective 변환 비활성화
        # flipud=0.0,  # 상하 뒤집기 비활성화
        # fliplr=0.2,  # 좌우 뒤집기 최소화
        # mosaic=0.0,  # mosaic 완전 비활성화
        # hsv_h=0.0,  # HSV 변화 비활성화
        # hsv_s=0.0,  # HSV 채도 변화 비활성화
        # hsv_v=0.0,  # HSV 명도 변화 비활성화
        # auto_augment=None,  # 자동 증강 비활성화
        # erasing=0.0,  # random erasing 비활성화
        # crop_fraction=1.0,  # crop 비활성화
        
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect",
        name=f"{model_nm}",
        verbose=True,  # 자세한 로그 출력
    )

best_weight_path = (
    f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/best.pt"
)

# print("🎉 안전 모드 학습 완료!")
print(f"Best 가중치 저장 경로: {best_weight_path}")

# 나머지 코드는 학습 완료 후 활성화
if DO_VAL or DO_PREDICT:
    if not os.path.exists(best_weight_path):
        print(f"[ERROR] best.pt 파일이 존재하지 않습니다: {best_weight_path}")
        exit(1)
    model_Product = YOLO(best_weight_path)


if DO_VAL:
    metrics = model_Product.val()
    print(metrics)
    pass


# 예측 후처리 수행을 위한 변수 설정
pre_cls = 1.0  # 클래스 수 1개 이상
pre_kobj = 1.0  # 객체 수 1개 이상
pred_conf = 0.5  # 신뢰도 0.5 이상
pred_dfl = 1  # DFL 사용 여부 (1: 사용, 0: 미사용) 


if DO_PREDICT:
    # 모델 예측 수행
    results = model_Product.predict(
        f"/hdd/datasets/dod_data/{Product}/{data_nm}/val/images",
        save=True,
        save_crop=True,
        save_txt=True,
        cls=pre_cls,
        kobj=pre_kobj,
        conf=pred_conf,
        dfl=pred_dfl,
        exist_ok=True, # 기존 결과 덮어쓰기
        project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
        name=f"pred_{model_nm}_val",
        # name=f"pred_{model_nm}_val_finetune",  # 기존과 다른 이름으로 지정
    )

    ### 신뢰도 없이 라벨만 show_conf = False
    # show_conf=False는 결과 이미지에만 적용됨. save_txt=True로 저장되는 txt에는 conf가 항상 포함됨.
    # txt에서 conf를 제거하려면 별도 후처리 필요.
    results = model_Product.predict(
        f"/hdd/datasets/dod_data/{Product}/{data_nm}/val/images",
        save=True,
        save_crop=True,
        save_txt=True,
        show_conf=False,  # 이미지에 신뢰도 미표시 (txt에는 항상 conf 포함)
        cls=pre_cls,
        kobj=pre_kobj,
        conf=pred_conf,
        dfl=pred_dfl,
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
        name=f"pred_{model_nm}_val_without_conf",
        # name=f"pred_{model_nm}_val_without_conf_finetune",  # 기존과 다른 이름으로 지정
    )

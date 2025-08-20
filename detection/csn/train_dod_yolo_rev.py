import os
import torch
import torch.hub
from ultralytics import YOLO
import gc

# 초기 설정
Date = "250820"
Time = "09h"  # 시간 업데이트
User = "hwoh"
Product = "csn"
yolo_nm = "11n"
model_nm = f"{Product}_dod_{yolo_nm}_{Date}{Time}"
data_nm = "v5"

os.chdir(f"/home/{User}/detection/{Product}")
torch.hub.set_dir(f"/home/{User}/detection/{Product}")

# YOLO 모델 로딩 - 완전 자동화
print(f"🔍 YOLO 모델 로딩: yolo{yolo_nm}.pt")

# Ultralytics가 알아서 처리 (로컬 확인 → 캐시 확인 → 다운로드)
model = YOLO(f"yolo{yolo_nm}.pt")

print(f"✅ 모델 준비 완료: yolo{yolo_nm}.pt")
print(f"📁 모델 경로: {model.ckpt_path if hasattr(model, 'ckpt_path') else '캐시됨'}")
        
# 모델 설정
DO_TRAIN = True
DO_VAL = False
DO_PREDICT = True

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
    
    #######################
    import cv2
    ####################################################################################
    ####################################################################################
    # 데이터셋 이미지/라벨 파일 점검 (확장자 자동 처리)
    import os
    train_txt_path = f"/hdd/datasets/dod_data/{Product}/{data_nm}/train.txt"
    missing_images = []
    missing_labels = []

    with open(train_txt_path) as f:
        for line in f:
            img_path = line.strip()
            # 이미지 확장자와 상관없이 .txt로 변경
            label_path = os.path.splitext(img_path)[0] + ".txt"
            label_path = label_path.replace("images", "labels")
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                print(f"[X] 이미지 로딩 실패: {img_path}")
                missing_images.append(img_path)
            if not os.path.exists(label_path):
                print(f"[X] 라벨 파일 없음: {label_path}")
                missing_labels.append(label_path)

    if missing_images or missing_labels:
        print(f"[ERROR] 손상 이미지: {len(missing_images)}개, 라벨 없음: {len(missing_labels)}개. 학습 중단.")
        exit(1)
    else:
        print("[OK] 모든 이미지/라벨 파일 정상.")
    ####################################################################################

# 학습 데이터 이미지 경로 사전 점검
train_txt_path = f"/hdd/datasets/dod_data/{Product}/{data_nm}/train.txt"
missing_files = []
with open(train_txt_path) as f:
    for line in f:
        img_path = line.strip()
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            print(f"[X] 이미지 로딩 실패: {img_path}")
            missing_files.append(img_path)

if missing_files:
    print(f"[ERROR] 총 {len(missing_files)}개의 이미지가 손상 또는 없음. 학습을 중단합니다.")
    exit(1)
else:
    print("[OK] 모든 학습 이미지가 정상적으로 로딩됩니다.")
    #######################
    
    # 학습 시작
    model.train(
        data=f"/hdd/datasets/dod_data/{Product}/{Product}_defect_detection_data_{data_nm}.yaml",
        epochs=500,
        batch=32,
        patience=150,
        dropout=0.22,
        iou=0.52, 
        lr0=0.0005,
        lrf=0.00001,
        optimizer="AdamW",
        workers=0, # 워커 수 설정
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect",
        name=f"{model_nm}",
        verbose=True,
    )

best_weight_path = (
    f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/best.pt"
)

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
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
        name=f"pred_{model_nm}_val",
    )

    ### 신뢰도 없이 라벨만 show_conf = False
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

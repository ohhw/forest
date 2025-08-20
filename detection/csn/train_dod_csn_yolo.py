from ultralytics import YOLO
import os
import torch.hub

# 초기 설정
Date = "250627"
Time = "10h"
User = "hwoh"
Product = "csn"
yolo_nm = "11n"
model_nm = f"{Product}_dod_{yolo_nm}_{Date}{Time}"

os.chdir(f"/home/{User}/detection/{Product}")
torch.hub.set_dir(f"/home/{User}/detection/{Product}")

# YOLO 가중치 경로 설정
model = YOLO(f"/home/{User}/detection/{Product}/yolo{yolo_nm}.pt")

# 모델 학습
model.train(
    data=f"/hdd/datasets/dod_data/{Product}/{Product}_defect_detection_data_v2.yaml",
    epochs=500,
    batch=32,
    patience=150,
    dropout=0.24,
    iou=0.54, 
    lr0=0.0005, # 초기 학습률
    lrf=0.00001, # 학습률 감소 비율
    optimizer="AdamW",
    exist_ok=True,
    project=f"/home/{User}/detection/{Product}/runs/detect",
    name=f"{model_nm}",
    # resume=True,  # 이전 학습 재개
    # device="0",   # <--- 이 줄 추가
)

# 텐서보드 안내 메시지 출력
print(f"[INFO] 텐서보드 실행: tensorboard --logdir /home/{User}/detection/{Product}/runs/detect")

# 베스트 모델 로드
best_weight_path = f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/best.pt"
if not os.path.exists(best_weight_path):
    best_weight_path = f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/last.pt"
    if not os.path.exists(best_weight_path):
        raise FileNotFoundError(f"가중치 파일이 존재하지 않습니다: {best_weight_path}")

model_Product = YOLO(best_weight_path)

# 예측 후처리 수행을 위한 변수 설정
# pre_cls = 1.0  # 클래스 수 1개 이상
# pre_kobj = 1.0  # 객체 수 1개 이상
# pred_conf = 0.5  # 신뢰도 0.5 이상
# pred_dfl = 1  # DFL 사용 여부 (1: 사용, 0: 미사용) 

# 모델 예측 수행
results = model_Product.predict(    
    f"/hdd/datasets/dod_data/{Product}/val2/images",
    save=True,
    save_crop=True,
    save_txt=True,
    # cls=pre_cls,
    # kobj=pre_kobj,
    # conf=pred_conf,
    # dfl=pred_dfl,
    exist_ok=True,
    project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
    name=f"pred_{model_nm}_val",
)

# 신뢰도 없이 라벨만 show_conf = False
results = model_Product.predict(
    f"/hdd/datasets/dod_data/{Product}/val2/images",
    save=True,
    save_crop=True,
    save_txt=True,
    # cls=pre_cls,
    # kobj=pre_kobj,
    # conf=pred_conf,
    # dfl=pred_dfl,
    show_conf=False,
    exist_ok=True,
    project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
    name=f"pred_{model_nm}_val_without_conf",
)
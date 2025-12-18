import os
import torch.hub
from ultralytics import YOLO


# 초기 설정
Date = "250715"
Time = "10h"
User = "hwoh"  # 사용자 홈 폴더 경로명
Product = "jjb"  # 임산물 이름 (예: jjb, wln, csn 등)
yolo_nm = "11l"  # YOLO 모델명 직접 지정 (예: 11s, 11n 등)
model_nm = f"test_{Product}_dod_{yolo_nm}_{Date}{Time}"  # 품종_dod_모델명_날짜시간
data_nm = "v10"  # 데이터 버전

os.chdir(f"/home/{User}/detection/{Product}")  # 작업 디렉토리 고정
torch.hub.set_dir(f"/home/{User}/detection/{Product}")

# YOLO 가중치 경로 설정
model = YOLO(f"/home/{User}/detection/{Product}/yolo{yolo_nm}.pt")  # 사전학습된 YOLO 가중치 경로
# model = YOLO(f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/best.pt")  # finetune된 모델 가중치 경로


# 모델 설정
DO_TRAIN = True
DO_VAL = False
DO_PREDICT = True


if DO_TRAIN:
    # 모델 학습
    model.train(
        data=f"/hdd/datasets/dod_data/{Product}/{Product}_defect_detection_data_{data_nm}.yaml",
        epochs=300, 
        batch=32, 
        patience=100, 
        dropout=0.3, 
        iou=0.3, 
        lr0=0.0005, # 초기 학습률
        lrf=0.00001, # 학습률 감소 비율
        optimizer="AdamW",
        exist_ok=True,
        project=f"/home/{User}/detection/{Product}/runs/detect",
        name=f"{model_nm}",
        # name=f"{model_nm}_finetune",  # 기존과 다른 이름으로 지정
    )
best_weight_path = (
    f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/best.pt"
)


if DO_VAL or DO_PREDICT:
    if not os.path.exists(best_weight_path):
        print(f"[ERROR] best.pt 파일이 존재하지 않습니다: {best_weight_path}")
        print("아래와 같은 원인일 수 있습니다:")
        print(" - 학습이 아직 완료되지 않아 best.pt가 생성되지 않았습니다.")
        print(" - 초기 설정값이 잘못되었습니다.")
        print(" - 경로에 오타가 있거나, 파일이 다른 위치에 저장되었습니다.")
        print(" - 학습을 먼저 완료하거나, 경로 및 설정값을 다시 확인해 주세요.")
        exit(1)  # 또는 raise FileNotFoundError
    model_Product = YOLO(best_weight_path)


if DO_VAL:
    metrics = model_Product.val(
        device=0,)
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

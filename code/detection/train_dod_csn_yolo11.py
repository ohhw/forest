from ultralytics import YOLO

# 정확하게 작성시 YOLO 모델명 추출까지 진행됨
model_nm = "csn_dod_11n_25041017h"  # 품종_dod_사전학습가중치_날짜시간

# YOLO 모델 이름 추출 (예: 11s, 11n 등)
yolo_model = model_nm.split('_')[2]  # model_nm의 세 번째 요소 추출

# YOLO 가중치 경로 설정
model = YOLO(f'/home/hwoh/detection/csn/yolo{yolo_model}.pt')  # 사전학습 가중치 변경

# 모델 학습
model.train(
    data="/hdd/datasets/dod_data/csn/csn_defect_detection_data.yaml",
    epochs=150,
    batch=32,
    patience=50,
    # optimizer='SGD',
    dropout=0.25,
    iou=0.35,
    exist_ok=True,
    project=f"/home/hwoh/detection/csn/runs/detect",  # 수정된 경로
    name=f"{model_nm}",
)

# 베스트 모델 로드
model_csn = YOLO(f"/home/hwoh/detection/csn/runs/detect/{model_nm}/weights/best.pt")

# 모델 예측 수행
results = model_csn.predict(
    "/hdd/datasets/dod_data/csn/v2/val/images",
    save=True,
    save_crop=True,
    save_txt=True,
    project=f"/home/hwoh/detection/csn/runs/detect/{model_nm}",
    name=f"pred_{model_nm}_val",
)

### 신뢰도 없이 라벨만  show_conf = False
results = model_csn.predict(
    "/hdd/datasets/dod_data/csn/v2/val/images",
    save=True,
    save_crop=True,
    save_txt=True,
    show_conf=False,
    project=f"/home/hwoh/detection/csn/runs/detect/{model_nm}",
    name=f"pred_{model_nm}_val_without_conf",
)
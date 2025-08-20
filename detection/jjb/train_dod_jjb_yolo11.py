from ultralytics import YOLO

# 정확하게 작성시 YOLO 모델명 추출까지 진행됨
model_nm = "jjb_dod_11m_25060912h"  # 품종_dod_사전학습가중치_날짜시간

# YOLO 모델 이름 추출 (예: 11s, 11n 등)
yolo_model = model_nm.split('_')[2]  # model_nm의 세 번째 요소 추출

data_nm = "v9"  # 데이터 버전

# YOLO 가중치 경로 설정
model = YOLO(f'/home/hwoh/detection/jjb/yolo{yolo_model}.pt')  # 사전학습 가중치 변경

# 모델 학습
model.train(
    data=f"/hdd/datasets/dod_data/jjb/jjb_defect_detection_data_{data_nm}.yaml",
    epochs=100,
    batch=16,
    # optimizer='SGD',
    patience=50,
    dropout=0.2,
    iou=0.5,
    exist_ok=True,
    project=f"/home/hwoh/detection/jjb/runs/detect",  # 수정된 경로
    name=f"{model_nm}",
)

# 베스트 모델 로드
# 개인 폴더 경로로 수정 필요
model_jjb = YOLO(f"/home/hwoh/detection/jjb/runs/detect/{model_nm}/weights/best.pt")

# # # 베스트 모델 summary
# metrics = model_jjb.val()
# print(metrics)

# 모델 예측 수행
results = model_jjb.predict(f'/hdd/datasets/dod_data/jjb/{data_nm}/val/images', 
                            save=True, 
                            save_crop=True, 
                            save_txt=True,  
                            project=f"/home/hwoh/detection/jjb/runs/detect/{model_nm}", 
                            name=f'pred_{model_nm}_val',
)

### 신뢰도 없이 라벨만  show_conf = False
results = model_jjb.predict(f'/hdd/datasets/dod_data/jjb/{data_nm}/val/images', 
                            save=True, 
                            save_crop=True, 
                            save_txt=True, 
                            show_conf = False, 
                            exist_ok=True, 
                            project=f"/home/hwoh/detection/jjb/runs/detect/{model_nm}", 
                            name=f'pred_{model_nm}_val_without_conf'
)
from ultralytics import YOLO

# 정확하게 작성시 YOLO 모델명 추출까지 진행됨
model_nm = 'wln_dod_11s_25042313h'  # 품종_dod_사전학습가중치_날짜시간

# YOLO 모델 이름 추출 (예: 11s, 11n 등)
yolo_model = model_nm.split('_')[2]  # model_nm의 세 번째 요소 추출

# YOLO 가중치 경로 설정
model = YOLO(f'/home/hwoh/detection/wln/yolo{yolo_model}.pt')  # 사전학습 가중치 변경

# 모델 학습
results = model.train(
    data='/hdd/datasets/dod_data/wln/wln_defect_detection_data.yaml', 
    epochs=250, 
    batch=32,
    patience=100, 
    dropout=0.255, 
    iou=0.415, 
    exist_ok=True,
    project='/home/hwoh/detection/wln/runs/detect',  # 프로젝트 절대 경로
    name=f"{model_nm}",
)

# 학습된 모델 로드 -> 개인 폴더 경로로 수정 필요
model_wln = YOLO(f'/home/hwoh/detection/wln/runs/detect/{model_nm}/weights/best.pt')

# 모델 예측 수행
results = model_wln.predict('/hdd/datasets/dod_data/wln/val2/images', 
                            save=True, 
                            save_crop=True, 
                            save_txt=True,  
                            project=f"/home/hwoh/detection/wln/runs/detect/{model_nm}", 
                            name=f'pred_{model_nm}_val',
)

results = model_wln.predict('/hdd/datasets/dod_data/wln/val2/images', 
                            save=True, 
                            save_crop=True, 
                            save_txt=True, 
                            show_conf=False, 
                            exist_ok=True, 
                            project=f"/home/hwoh/detection/wln/runs/detect/{model_nm}", 
                            name=f'pred_{model_nm}_val_without_conf'
)
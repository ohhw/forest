## 임산물 색택 분류 모델을 위해 이미지 분류 사전학습 모델 활용 전이학습 진행
from ultralytics import YOLO
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# GPU 메모리 캐시 비우기 (메모리 부족 방지)
torch.cuda.empty_cache()

# 1. 변수 및 모델 준비
Date = '250211'  # 날짜
Time = '16h'     # 시간
User = 'hwoh'
Product = 'jjb'
yolo_nm = 'yolo11s-cls'  # YOLO 모델 이름
model_nm = f'{Product}_cls_{yolo_nm}{Date}{Time}'  # 학습 결과 저장에 사용할 모델 이름
pred_nm = f'pred_{model_nm}'  # 예측 결과 저장에 사용할 이름

os.chdir(f'/home/{User}/classification/{Product}')  # 작업 디렉토리 고정
torch.hub.set_dir(f'/home/{User}/classification/{Product}')  # Torch 허브 경로 설정

# 사전학습된 YOLO 분류 모델 로드 (yolo11s-cls.pt 파일 필요)
model = YOLO(f'/home/{User}/classification/{Product}/{yolo_nm}.pt')

# 학습 시작 시간 기록
print('start')
start_time = time.time()

# 2. 모델 학습
model.train(
    data='/hdd/datasets/cls_data/jjb',  # 학습 데이터셋 경로
    epochs=100,                         # 학습 반복 횟수
    imgsz=224,                          # 입력 이미지 크기
    batch=128,                          # 배치 크기
    name=model_nm,                      # 결과 저장 폴더명
)

end_time = time.time()  # 학습 종료 시간 기록
training_time = end_time - start_time   # 총 학습 시간 계산
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)
print(f'학습 시간: {hours} : {minutes} : {seconds}')

# 3. 학습된 모델 로드
# 학습이 끝난 후 저장된 best.pt(최적 가중치) 모델을 다시 로드
model = YOLO(f'/home/{User}/classification/{Product}/runs/classify/{model_nm}/weights/best.pt')

# 4. 검증 데이터셋 준비
val_dir = '/hdd/datasets/cls_data/jjb/validation'  # 검증 데이터셋 경로
image_paths = []  # 검증 이미지 파일 경로 리스트
y_true = []       # 실제 클래스 인덱스 리스트
class_names = sorted(os.listdir(val_dir))  # 클래스 이름을 정렬하여 일관성 유지

# 각 클래스 폴더에서 이미지 경로와 실제 라벨 수집
img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.ppm')
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(val_dir, class_name)
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(img_exts):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            y_true.append(class_idx)

# 5. 모델 예측
# 검증 이미지 전체에 대해 예측 수행 (결과 저장)
results = model.predict(image_paths, save=True, save_txt=True, name=pred_nm)

# 6. 예측 결과 처리
y_pred = []  # 예측 클래스 인덱스 리스트
for result in results:
    # probs 속성이 있으면 분류(top1) 결과 사용
    if hasattr(result, 'probs'):
        y_pred.append(result.probs.top1)
    else:
        # detection 결과라면 가장 높은 신뢰도의 클래스 사용
        max_conf_class = result.boxes.cls[result.boxes.conf.argmax()].item()
        y_pred.append(int(max_conf_class))

# 7. 성능 평가 및 시각화
# 혼동 행렬(Confusion Matrix) 생성 및 출력
cm = confusion_matrix(y_true, y_pred)
print(f'Confusion Matrix for {yolo_nm}: ')
print(cm)

# 혼동 행렬 시각화 및 파일로 저장
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(f'/home/{User}/classification/{Product}/runs/classify/{model_nm}/confusion_matrix_viz.png')
plt.close()

# 분류 리포트(정밀도, 재현율 등) 출력
report = classification_report(y_true, y_pred, target_names=class_names)
print(f'Classification Report for {yolo_nm}: ')
print(report)
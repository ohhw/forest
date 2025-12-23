## 임산물(밤) 색택 분류 모델을 위해 이미지 분류 사전학습 모델 활용 전이학습 진행

from ultralytics import YOLO
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import time

# GPU cache
torch.cuda.empty_cache()

model_nm = 'jjb_25021116h'
pred_nm = f'pred_{model_nm}'

# load a pretrained YOLO11s
model = YOLO("yolo11s-cls.pt")
print('start')
start_time = time.time()

model.train(
    data='/hdd/datasets/cls_data/jjb',
    epochs=100,
    imgsz=224,
    batch=128,
    name=model_nm,
)
# results=model.val(data='/home/jmchoi/data_csn_cls_aws.yaml', save=True)

end_time = time.time()

# 총 학습한 시간 확인
training_time = end_time - start_time
# 초를 시간, 분, 초로 변환
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60) 
seconds = int(training_time % 60)

# 시간:분:초 형태로 출력
print(f"학습 시간: {hours} : {minutes} : {seconds}")

#### 모델 검증 시간 ####
# 학습된 모델 로드
# 개인 폴더 경로로 수정 필요
model = YOLO(f'/home/shoh/classification/classify/{model_nm}/weights/best.pt')

# 검증 데이터셋 경로
val_dir = '/hdd/datasets/cls_data/jjb/validation'

# 결과를 저장할 디렉토리 생성
#results_dir = 'evaluation_results'
#os.makedirs(results_dir, exist_ok=True)

# 이미지 경로와 실제 라벨 수집
image_paths = []
y_true = []
class_names = sorted(os.listdir(val_dir)) # 클래스 이름을 정렬하여 일관성 유지

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(val_dir, class_name)
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            y_true.append(class_idx) # 클래스 인덱스 사용

# 모델 예측 수행
results = model.predict(image_paths, save=True, save_txt=True, name=pred_nm)

# 예측 결과 수집
y_pred = []

for result in results:
    # probs 속성이 존재하는지 확인하고, 존재하면 argmax를 사용
    if hasattr(result, 'probs'):
        y_pred.append(result.probs.top1)  # top1은 가장 높은 확률을 가진 클래스의 인덱스
        
    else:
        # classification 작업이 아닌 경우 (예: detection)
        # 가장 높은 신뢰도를 가진 객체의 클래스로 대체
        max_conf_class = result.boxes.cls[result.boxes.conf.argmax()].item()
        y_pred.append(int(max_conf_class))

# Confusion Matrix 생성
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix for YOLO11s:')
print(cm)

# Confusion Matrix 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# 그림을 파일로 저장
# # 개인 폴더 경로로 수정 필요
plt.savefig(f'/home/shoh/classification/classify/{model_nm}/confusion_matrix_viz.png')
plt.close() # 메모리에서 현재 그림 삭제, 그림 창 닫기

# Classification Report 생성
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report for YOLO11s : ")
print(report)
import cv2
import os
from pathlib import Path

def plot_labels(image_path, label_path, class_names):
    # 이미지 불러오기
    img = cv2.imread(str(image_path))  # Path 객체를 문자열로 변환
    height, width, _ = img.shape
    
    # 라벨 파일 읽기
    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        # 클래스 ID를 정수로 변환하고 나머지 값을 소수로 변환
        class_id, x_center, y_center, box_width, box_height = map(float, label.split())
        class_id = int(class_id)  # 클래스 ID를 정수로 변환
        
        # YOLO 형식의 좌표를 이미지 좌표로 변환
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        
        # 바운딩 박스 좌상단과 우하단 좌표 계산
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # 바운딩 박스 그리기
        color = (0, 255, 0)  # 초록색
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_name = class_names[class_id]
        cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img

def save_label_images(image_folder, label_folder, output_dir, class_names):
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지와 라벨 파일 리스트 생성
    image_paths = sorted(image_folder.glob('*.bmp'))
    label_paths = sorted(label_folder.glob('*.txt'))

    # 이미지와 라벨 경로에 따라 라벨링된 이미지 저장
    for img_path, lbl_path in zip(image_paths, label_paths):
        # 라벨링 이미지 생성
        labeled_image = plot_labels(img_path, lbl_path, class_names)

        if labeled_image is None:
            print(f"Skipping saving due to error in image processing: {img_path}")
            continue

        # 저장할 파일명 설정
        img_name = img_path.stem + '_labeled.bmp'  # 파일명 변경
        save_path = output_dir / img_name
        
        # 이미지 저장
        cv2.imwrite(str(save_path), labeled_image)
        print(f'Saved labeled image: {save_path}')

# csn defect_top_rot
# image_folder = Path('C:\\Users\\USER\\Desktop\\data\\defect_detection\\aug\\csn_top_rot\\output\\images')
# label_folder = Path('C:\\Users\\USER\\Desktop\\data\\defect_detection\\aug\\csn_top_rot\\output\\labels')
# output_dir = Path('C:\\Users\\USER\\Desktop\\data\\defect_detection\\aug\\csn_top_rot\\viz\\labels')
# class_names = ['defect1_bug', 'defect2_crack', 'defect3_rot']

# jjb defect
image_folder = Path('/hdd/datasets/dod_data/jjb/v10/val/images')
label_folder = Path('/hdd/datasets/dod_data/jjb/v10/val/labels')
output_dir = Path('/home/hwoh/(temp)ground_truth')
class_names = ['defect1_overripe', 'defect2_crack', 'defect3_moldy', 'defect4_damage']

# 라벨링된 이미지 저장
save_label_images(image_folder, label_folder, output_dir, class_names)

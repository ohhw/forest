### 이미지 데이터 수량이 적어 기본적인 모양 변형 데이터 증강
### 데이터 증강 라이브러리인 imgaug 사용하여 임산물 고유의 모형과 색에 영향을 주지 않을 정도로만 변형

# 필요 라이브러리 import
import cv2
import numpy as np
import imgaug.augmenters as iaa  # 데이터 증강 메인 라이브러리
import imgaug as ia  # 데이터 증강 메인 라이브러리
import imageio # chatGPT 배경 흰색 코드에서 추가
from pathlib import Path

# 어노테이션 읽는 함수
def read_annotation(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    annotation = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        annotation.append([class_id, x_center, y_center, width, height])
    return annotation

# 어노테이션 저장 함수
def write_annotation(txt_file, annotation):
    with open(txt_file, 'w') as f:
        for box in annotation:
            class_id, x_center, y_center, width, height = box
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# imgaug에서 사용하는 어노테이션 표기에 맞게 기존 YOLO 어노테이션을 절대좌표로 변환하는 함수
def augment_and_save(image, annotation, output_image_path, output_annotation_path, seq):
    # imgaug는 절대좌표를 사용하므로 상대좌표를 절대좌표로 변환
    h, w = image.shape[:2]
    bboxes = []
    for box in annotation:
        class_id, x_center, y_center, bbox_width, bbox_height = box
        abs_x = x_center * w
        abs_y = y_center * h
        abs_w = bbox_width * w
        abs_h = bbox_height * h
        x_min = abs_x - abs_w / 2
        y_min = abs_y - abs_h / 2
        x_max = abs_x + abs_w / 2
        y_max = abs_y + abs_h / 2
        bboxes.append(ia.BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=class_id))

    # 증강 적용 코드
    bbs = ia.BoundingBoxesOnImage(bboxes, shape=image.shape)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

    # 다시 상대 좌표로 변환
    transformed_annotation = []
    for bbox in bbs_aug.bounding_boxes:
        x_min, y_min, x_max, y_max = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        abs_x = (x_min + x_max) / 2
        abs_y = (y_min + y_max) / 2
        abs_w = x_max - x_min
        abs_h = y_max - y_min
        x_center = abs_x / w
        y_center = abs_y / h
        bbox_width = abs_w / w
        bbox_height = abs_h / h
        transformed_annotation.append([bbox.label, x_center, y_center, bbox_width, bbox_height])

    cv2.imwrite(output_image_path, image_aug) # 아웃풋 이미지 저장

    success = cv2.imwrite(output_image_path, image_aug) 
    if not success:
        print(f"Failed to save image: {output_image_path}")


    write_annotation(output_annotation_path, transformed_annotation) # 아웃풋 어노테이션 저장

######################################################        폴더 경로 설정         #######################################################################################
# # jjb aug 경로 설정
# image_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_jjb/train/images') # 크롭된 이미지들이 저장된 폴더 경로
# annotation_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_jjb/train/labels') # 크롭된 어노테이션 파일들이 저장된 폴더 경로
# output_image_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_jjb/train_aug/images') # 증강된 이미지들이 저장될 폴더 경로
# output_annotation_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_jjb/train_aug/labels') # 증강된 어노테이션 파일들이 저장될 폴더 경로

# # wln aug 경로 설정
# image_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_wln/train/images') # 크롭된 이미지들이 저장된 폴더 경로
# annotation_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_wln/train/labels') # 크롭된 어노테이션 파일들이 저장된 폴더 경로
# output_image_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_wln/train_aug/images') # 증강된 이미지들이 저장될 폴더 경로
# output_annotation_folder = Path('F:/yolo_dataset/2024_defect_detection/crop_annotation/crop_wln/train_aug/labels') # 증강된 어노테이션 파일들이 저장될 폴더 경로

# # wln aug 경로 설정(24년 7월 이후 순서, 로컬 -> GPU서버)
# image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/wln/train_origin/images')
# annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/wln/train_origin/labels')
# output_image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/wln/train_aug_bgwhite/images')
# output_annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/wln/train_aug_bgwhite/labels')

# # jjb aug 경로 설정(24년 7월 이후, 로컬 -> GPU서버)
# image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/crop_annotation-/crop_jjb/train/images')
# annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/crop_annotation-/crop_jjb/train/labels')
# output_image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/jjb/train_aug_bgwhite/images')
# output_annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/jjb/train_aug_bgwhite/labels')

# jjb 예전에 했던걸로 aug 경로 설정(24년 7월 이후, 로컬 -> GPU서버)
# image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/crop_annotation/crop_jjb/train/images')
# annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/crop_annotation/crop_jjb/train/labels')
# output_image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/jjb/train_aug_bgwhite/images')
# output_annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/jjb/train_aug_bgwhite/labels')

# csn val defect2_crack 위주 aug 경로 설정
# image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/val/crack_i')
# annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/val/crack_l')
# output_image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/val_aug_crack/images')
# output_annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/val_aug_crack/labels')

# csn val defect2_crack 위주 aug 경로 설정
# image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/train/images')
# annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/train/labels')
# output_image_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/train_aug/images')
# output_annotation_folder = Path('/home/jmchoi/datasets/2024_defect_detection/yolo/csn/train_aug/labels')

# wnl val defect2_crack 위주 aug 경로 설정
# image_folder = Path('C:\\Users\\USER\\data\\defect_detection\\aug\\wln_crack\\images')
# annotation_folder = Path('C:\\Users\\USER\\data\\defect_detection\\aug\\wln_crack\\labels')
# output_image_folder = Path('C:\\Users\\USER\\data\\defect_detection\\aug\\wln_crack\\output\\images')
# output_annotation_folder = Path('C:\\Users\\USER\\data\\defect_detection\\aug\\wln_crack\\output\\labels')

# csn defect_top_rot 위주 aug 경로 설정
image_folder = Path('C:\\Users\\USER\\Desktop\\data\\detection\\dod_data\\jjb\\type_cls\\images\\1')
annotation_folder = Path('C:\\Users\\USER\\Desktop\\data\\detection\\dod_data\\jjb\\type_cls\\labels\\1')
output_image_folder = Path('C:\\Users\\USER\\Desktop\\data\\detection\\dod_data\\jjb\\type_cls\\output\\images\\1')
output_annotation_folder = Path('C:\\Users\\USER\\Desktop\\data\\detection\\dod_data\\jjb\\type_cls\\output\\labels\\1')

######################################################        폴더 경로 설정         #######################################################################################

# 출력 폴더 생성
output_image_folder.mkdir(parents=True, exist_ok=True)
output_annotation_folder.mkdir(parents=True, exist_ok=True)

# imgaug 증강 파이프라인 설정
seq = iaa.Sequential([
    # iaa.Sometimes(0.5, ( # 50% 확률로 랜덤 적용
    # iaa.Affine(scale=(0.9, 1.0), rotate=(-10, 10), cval=255), # scale : 크기 조절, cval : 배경 흰색(255) 단일값으로 설정
    iaa.Fliplr(1.0), # 좌우 반전
    iaa.Flipud(1.0), # 상하 반전
    # iaa.GaussianBlur(sigma=(0.0 , 1.0)), # 가우시안 블러
    # iaa.Multiply((0.8, 1.2)), # 밝기 조절
    # iaa.LinearContrast((0.75, 1.5)), # 대비 조절
    # iaa.Rotate((-10, 10)) # 회전
])

# 모든 이미지 파일 처리
for image_path in image_folder.glob('*.bmp'):  # 현재 원본 bmp만 있음. 다른 포맷도 추가 가능
    print(f"Processing image: {image_path}")
    annotation_path = annotation_folder / (image_path.stem + '.txt')

    # 어노테이션 파일 없는 경우 파일 이름 출력
    if not annotation_path.exists():
        print(f"Annotation for {image_path} not found, skipping.")
        continue

    # 이미지 및 어노테이션 불러오기
    image = cv2.imread(str(image_path))
    annotation = read_annotation(annotation_path)

    # 증강 및 저장
    for i in range(1):  # 각 이미지에 대해 ()개의 증강된 이미지 생성
        output_image_path = output_image_folder / f"{image_path.stem}_aug_{i}.bmp"
        output_annotation_path = output_annotation_folder / f"{annotation_path.stem}_aug_{i}.txt"
        augment_and_save(image, annotation, str(output_image_path), str(output_annotation_path), seq)
        print(f"Processed {output_image_path} and saved results.")

print('finish')
"""
YOLO 데이터셋 증강 클래스
YOLODatasetManager와 함께 사용
"""

import cv2
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import random

class YOLODataAugmenter:
    def __init__(self):
        # 기본 증강 설정
        self.augmentation_configs = {
            "기본": self._get_basic_augmenter(),
            "밝기": self._get_brightness_augmenter(),
            "회전": self._get_rotation_augmenter(),
            "노이즈": self._get_noise_augmenter(),
            "흐림": self._get_blur_augmenter(),
            "색상": self._get_color_augmenter(),
        }
    
    def _get_basic_augmenter(self):
        return iaa.Sequential([
            iaa.Sometimes(0.5, iaa.flip.Fliplr()),
            iaa.Sometimes(0.5, iaa.flip.Flipud()),
            iaa.Sometimes(0.5, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                mode='constant',
                cval=255
            ))
        ])
    
    def _get_brightness_augmenter(self):
        return iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30))
        ])
    
    def _get_rotation_augmenter(self):
        return iaa.Sequential([
            iaa.Affine(rotate=(-45, 45), mode='constant', cval=255)
        ])
    
    def _get_noise_augmenter(self):
        return iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
        ])
    
    def _get_blur_augmenter(self):
        return iaa.Sequential([
            iaa.GaussianBlur(sigma=(0.0, 1.0))
        ])
    
    def _get_color_augmenter(self):
        return iaa.Sequential([
            iaa.AddToHueAndSaturation((-20, 20)),
            iaa.LinearContrast((0.8, 1.2))
        ])

    def read_annotation(self, txt_file: str) -> List[List[float]]:
        """YOLO 형식의 어노테이션 파일 읽기"""
        if not os.path.exists(txt_file):
            return []
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        annotation = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            annotation.append([class_id, x_center, y_center, width, height])
        return annotation

    def write_annotation(self, txt_file: str, annotation: List[List[float]]):
        """YOLO 형식의 어노테이션 파일 쓰기"""
        with open(txt_file, 'w') as f:
            for box in annotation:
                class_id, x_center, y_center, width, height = box
                f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

    def augment_and_save(self, 
                        image: np.ndarray,
                        annotation: List[List[float]],
                        output_image_path: str,
                        output_annotation_path: str,
                        augmenter: iaa.Sequential) -> bool:
        """이미지와 어노테이션을 함께 증강하고 저장"""
        try:
            # 이미지 크기 가져오기
            h, w = image.shape[:2]
            
            # YOLO 좌표를 절대 좌표로 변환
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

            # 증강 적용
            bbs = ia.BoundingBoxesOnImage(bboxes, shape=image.shape)
            image_aug, bbs_aug = augmenter(image=image, bounding_boxes=bbs)
            bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

            # 증강된 좌표를 YOLO 형식으로 변환
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

            # 결과 저장
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_annotation_path), exist_ok=True)
            
            success = cv2.imwrite(output_image_path, image_aug)
            if not success:
                print(f"이미지 저장 실패: {output_image_path}")
                return False
            
            self.write_annotation(output_annotation_path, transformed_annotation)
            return True
            
        except Exception as e:
            print(f"증강 중 오류 발생: {str(e)}")
            return False

    def augment_dataset(self,
                       image_dir: str,
                       label_dir: str,
                       output_image_dir: str,
                       output_label_dir: str,
                       augmentation_types: List[str],
                       copies_per_image: int = 1,
                       progress_callback=None) -> Tuple[int, int]:
        """데이터셋 전체 증강
        
        Args:
            image_dir: 원본 이미지 디렉토리
            label_dir: 원본 라벨 디렉토리
            output_image_dir: 증강된 이미지 저장 디렉토리
            output_label_dir: 증강된 라벨 저장 디렉토리
            augmentation_types: 적용할 증강 유형 리스트 ["기본", "밝기", "회전" 등]
            copies_per_image: 이미지당 생성할 증강 복사본 수
            progress_callback: 진행상황 콜백 함수 (현재 수, 전체 수) -> None
            
        Returns:
            (성공 수, 실패 수) 튜플
        """
        # 출력 디렉토리 생성
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # 이미지 파일 목록 가져오기
        image_files = []
        for ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        total_files = len(image_files) * copies_per_image * len(augmentation_types)
        processed = 0
        success_count = 0
        failed_count = 0
        
        # 각 이미지에 대해 증강 수행
        for img_path in image_files:
            # 원본 이미지와 라벨 읽기
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"이미지를 읽을 수 없음: {img_path}")
                failed_count += copies_per_image * len(augmentation_types)
                continue
            
            label_path = Path(label_dir) / f"{img_path.stem}.txt"
            annotation = self.read_annotation(str(label_path))
            
            # 각 증강 유형별로 처리
            for aug_type in augmentation_types:
                augmenter = self.augmentation_configs.get(aug_type)
                if augmenter is None:
                    print(f"알 수 없는 증강 유형: {aug_type}")
                    continue
                
                # 복사본 생성
                for copy_idx in range(copies_per_image):
                    # 출력 파일 경로 생성
                    output_image_name = f"{img_path.stem}_{aug_type}_{copy_idx}{img_path.suffix}"
                    output_image_path = Path(output_image_dir) / output_image_name
                    output_label_path = Path(output_label_dir) / f"{img_path.stem}_{aug_type}_{copy_idx}.txt"
                    
                    # 증강 수행
                    if self.augment_and_save(image, annotation, 
                                           str(output_image_path), 
                                           str(output_label_path),
                                           augmenter):
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_files)
        
        return success_count, failed_count

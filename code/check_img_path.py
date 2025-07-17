import cv2
import glob

img_paths = glob.glob('/hdd/datasets/dod_data/jjb/250709_add_data/images/*')
for path in img_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"이미지 읽기 실패: {path}")
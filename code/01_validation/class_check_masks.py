# library
import cv2
import numpy as np
import glob

# directory
mask_files = glob.glob('/hdd/datasets/seg_data/psm4/train/masks/*.png')  # 경로에 있는 모든 PNG 파일

all_labels = set()
for mf in mask_files:
    mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
    unique_labels = np.unique(mask)
    all_labels.update(unique_labels)

# print
print("전체 데이터셋에 등장하는 클래스(픽셀값):", sorted(list(all_labels)))









"""
YOLO 모델 학습에 필요한 데이터 경로 텍스트 파일 생성 스크립트
2025.05.15 SSD 이슈로 코드 테스트 및 현행화
"""

from glob import glob

# 설정
product = "jjb"
data_ver = "v8"
img_exts = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "BMP",
    "JPG",
    "JPEG",
    "PNG",
    "TIF",
    "TIFF",
]  # 지원 확장자

# 이미지 경로 리스트 생성 (확장자별로 추가)
train_img_list = []
for ext in img_exts:
    # v8 디렉토리 이미지
    train_img_list.extend(glob(f"/hdd/datasets/dod_data/{product}/{data_ver}/train/images/*.{ext}"))
    # 250529 추가 데이터
    train_img_list.extend(glob(f"/hdd/datasets/dod_data/{product}/250529_add_data/images/*.{ext}"))
    # # 250916 추가 데이터
    # train_img_list.extend(glob(f"/hdd/datasets/dod_data/{product}/250916_add_data/images/*.{ext}"))
"""
# valid 이미지 경로 리스트 생성 (확장자별로 추가)
valid_img_list = []
for ext in img_exts:
    valid_img_list += glob(
        f"/hdd/datasets/dod_data/{product}/{data_ver}/val/images/*.{ext}"
    )

# test 이미지 경로 리스트 생성 (확장자별로 추가)
test_img_list = []
for ext in img_exts:
    test_img_list += glob(
        f"/hdd/datasets/dod_data/{product}/{data_ver}/test/images/*.{ext}"
    )
"""
# train 리스트를 txt 파일로 저장
with open(f"/hdd/datasets/dod_data/{product}/v9/train.txt", "w") as f:
    f.write("\n".join(train_img_list) + "\n")
"""
# valid 리스트를 txt 파일로 저장
with open(f"/hdd/datasets/dod_data/{product}/v9/valid.txt", "w") as f:
    f.write("\n".join(valid_img_list) + "\n")

# test 리스트를 txt 파일로 저장
with open(f"/hdd/datasets/dod_data/{product}/{data_ver}/test.txt", "w") as f:
    f.write("\n".join(test_img_list) + "\n")
"""

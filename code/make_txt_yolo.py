
"""
YOLO 모델 학습에 필요한 데이터 경로 텍스트 파일 생성 스크립트
2025.05.15 SSD 이슈로 코드 테스트 및 현행화
"""

from glob import glob

"""# 설정
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


"""
# train 리스트를 txt 파일로 저장
with open(f"/hdd/datasets/dod_data/{product}/v9/train.txt", "w") as f:
    f.write("\n".join(train_img_list) + "\n")
"""

"""
# valid 리스트를 txt 파일로 저장
with open(f"/hdd/datasets/dod_data/{product}/v9/valid.txt", "w") as f:
    f.write("\n".join(valid_img_list) + "\n")

# test 리스트를 txt 파일로 저장
with open(f"/hdd/datasets/dod_data/{product}/{data_ver}/test.txt", "w") as f:
    f.write("\n".join(test_img_list) + "\n")
"""


################################################################################################################################################################
# obj_data용 경로 txt 파일 생성
# 설정
product = "obj"
data_nm = "251015_psm"
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
    # 251015_psm 데이터 (실제 경로: /hdd/datasets/obj_data/251015_psm/images)
    train_img_list.extend(glob(f"/hdd/datasets/{product}_data/{data_nm}/images/*.{ext}"))
    
# train 리스트를 txt 파일로 저장
out_txt = f"/hdd/datasets/{product}_data/{data_nm}/train.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(train_img_list) + ("\n" if train_img_list else ""))

print(f"[make_txt_yolo] images found: {len(train_img_list)}")
print(f"[make_txt_yolo] written: {out_txt}")

"""
# 참고: valid/test도 train과 동일 경로(images)를 사용하여 생성하려면 아래 주석을 해제하세요.

# valid 리스트 저장
# valid_txt = f"/hdd/datasets/{product}_data/{data_nm}/valid.txt"
# with open(valid_txt, "w") as f:
#     f.write("\n".join(train_img_list) + ("\n" if train_img_list else ""))
# print(f"[make_txt_yolo] written: {valid_txt}")

# test 리스트 저장
# test_txt = f"/hdd/datasets/{product}_data/{data_nm}/test.txt"
# with open(test_txt, "w") as f:
#     f.write("\n".join(train_img_list) + ("\n" if train_img_list else ""))
# print(f"[make_txt_yolo] written: {test_txt}")
"""
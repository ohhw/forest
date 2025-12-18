### from_path에 directory_path와 같은 이름의 파일이 존재하면 target_path로 이동하는 코드 ###

import os
import shutil

# 서버 내 특정 디렉터리 경로
directory_path = "/hdd/datasets/dod_data/jjb/v1_before250204/val/labels"
# images(train) 폴더에 일부 같은 이름이 있다면
from_path = "/home/shoh/classification/datasets/jjb/250204_jjb_v3_re/train/labels"
# val 폴더로 이동
target_path = "/home/shoh/classification/datasets/jjb/250204_jjb_v3_re/val/labels"

# target_path 없으면 생성
os.makedirs(target_path, exist_ok=True)

# (확인용) 파일 개수 세기
file_count = len([f for f in os.listdir(from_path) if os.path.isfile(os.path.join(from_path, f))])
print("images 파일 개수 :", file_count)

# directory_path에 있는 파일 이름 가져온 후 난수 제거
directory_files = set()
for file in os.listdir(directory_path):
    clean_filename = "-".join(file.split("-")[1:])  # 난수 제거
    clean_filename = os.path.splitext(clean_filename)[0]  # 확장자 제거
    directory_files.add(clean_filename)  # 정리된 파일명 저장

# from_path에 있는 파일 중 동일한 이름이 존재하는지 확인
for file in os.listdir(from_path):
    original_filename = "-".join(file.split("-")[1:])  # 난수 제거
    original_filename = os.path.splitext(original_filename)[0]  # 확장자 제거

    if original_filename in directory_files:
        # (확인용)
        print(f"같은 이름의 파일: {file}")
        src_file = os.path.join(from_path, file)
        dst_file = os.path.join(target_path, file)

        # 파일 이동
        shutil.move(src_file, dst_file)
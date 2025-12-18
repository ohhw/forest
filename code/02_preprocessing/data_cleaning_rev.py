# 실행 방법: python data_cleaning_rev.py /path/to/train.txt
# train.txt에는 각 줄마다 이미지 경로가 작성되어 있어야 합니다
# 예시:
# /data/gosari/images/img1.jpg
# /data/doraji/images/img2.jpg
# /data/chinnamul/images/img3.jpg

import os
import sys
import glob
import shutil
import datetime
from datetime import timezone
import pytz

# ===== 수정이 필요한 부분 =====
# train.txt 파일 경로를 지정하세요
TRAIN_TXT_PATH = "/hdd/datasets/dod_data/jjb/v11/train.txt"

# 지원되는 이미지 확장자
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# ===== 자동 설정 부분 (수정 불필요) =====
BASE_DIR = "/hdd/rollback"  # 작업 결과물이 저장될 기본 디렉토리
KST = pytz.timezone('Asia/Seoul')
BACKUP_TIME = datetime.datetime.now(KST).strftime('%Y%m%d_%H%M%S')

# 작업 경로 설정
PATHS = {
    'backup': f"{BASE_DIR}/backup_{BACKUP_TIME}",  # 백업 디렉토리
    'empty_list': f"{BASE_DIR}/list/empty_files_{BACKUP_TIME}.txt",  # 빈 파일 리스트
    'result': f"{BASE_DIR}/list/result_{BACKUP_TIME}.txt"  # 최종 결과 리포트
}

# *****************************************************************************
# 데이터 백업
def backup_data(train_txt_path):
    """원본 데이터 백업"""
    print("\nStarting backup process...")
    try:
        # train.txt 파일 읽기
        with open(train_txt_path, 'r') as f:
            image_paths = f.read().splitlines()

        if not image_paths:
            print("Error: train.txt is empty!")
            sys.exit(1)

        # 첫 번째 이미지 경로를 기준으로 디렉토리 구조 파악
        first_img = image_paths[0]
        img_dir = os.path.dirname(first_img)
        base_dir = os.path.dirname(img_dir)
        
        # 백업 디렉토리 생성
        os.makedirs(PATHS['backup'], exist_ok=True)
        backup_images = os.path.join(PATHS['backup'], 'images')
        backup_labels = os.path.join(PATHS['backup'], 'labels')
        os.makedirs(backup_images, exist_ok=True)
        os.makedirs(backup_labels, exist_ok=True)

        # train.txt 백업
        shutil.copy2(train_txt_path, PATHS['backup'])
        
        # images와 labels 디렉토리 백업
        orig_images = os.path.join(base_dir, 'images')
        orig_labels = os.path.join(base_dir, 'labels')
        
        if os.path.exists(orig_images):
            shutil.copytree(orig_images, backup_images, dirs_exist_ok=True)
        if os.path.exists(orig_labels):
            shutil.copytree(orig_labels, backup_labels, dirs_exist_ok=True)
            
        print(f"Backup completed successfully to {PATHS['backup']}")
    except Exception as e:
        print(f"Error during backup: {e}")
        sys.exit(1)

# *****************************************************************************
# train.txt에서 이미지 경로를 읽고 대응하는 레이블 파일 검사
def check_empty_txt_files(train_txt_path):
    """빈 레이블 파일 검사"""
    empty_files = []
    
    print("\nProcessing train.txt file...")
    with open(train_txt_path, 'r') as f:
        image_paths = f.read().splitlines()
    
    print(f"Found {len(image_paths)} image paths in train.txt")
    print("\nChecking label files...")
    
    for img_path in image_paths:
        # 이미지 경로가 유효한지 확인
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
            
        # 이미지 경로에서 labels 경로 생성
        img_dir = os.path.dirname(img_path)
        base_dir = os.path.dirname(img_dir)
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # labels 디렉토리의 레이블 파일 경로 생성
        label_dir = os.path.join(base_dir, 'labels')
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # 레이블 파일이 존재하고 비어있는지 확인
        if os.path.exists(label_path):
            if os.path.getsize(label_path) == 0:
                print(f"Found empty label file: {label_path}")
                empty_files.append((img_path, label_path))
            else:
                # 파일 내용 확인
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if not content:  # 공백이나 줄바꿈만 있는 경우
                        print(f"Found label file with only whitespace: {label_path}")
                        empty_files.append((img_path, label_path))
        else:
            print(f"Warning: Label file not found for {img_path}")

    print(f"\nFound {len(empty_files)} empty label files")
    return empty_files

# *****************************************************************************
# 결과 저장
def save_empty_files_list(empty_files, output_path):
    """빈 레이블 파일 목록 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"검사 시간: {datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} (KST)\n")
        f.write(f"Train.txt 경로: {TRAIN_TXT_PATH}\n")
        f.write(f"발견된 빈 레이블 파일 수: {len(empty_files)}\n")
        f.write("\n=== 빈 레이블 파일 목록 ===\n")
        for img_path, label_path in empty_files:
            f.write(f"Image: {img_path}\tLabel: {label_path}\n")

# *****************************************************************************
# 빈 레이블 파일에 대응하는 이미지 삭제
def remove_empty_label_images(empty_files):
    """불필요한 이미지 파일 삭제"""
    for img_path, _ in empty_files:
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                print(f"Removed image: {img_path}")
            except Exception as e:
                print(f"Error removing {img_path}: {e}")

# *****************************************************************************
# 이미지와 레이블 파일 수 확인
def verify_image_label_count(image_dir, label_dir):
    """이미지와 레이블 파일 수 검증"""
    image_count = len([f for f in os.listdir(image_dir) if f.endswith(IMG_EXTENSIONS)])
    label_count = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    
    print(f"\nVerification Results:")
    print(f"Number of images: {image_count}")
    print(f"Number of labels: {label_count}")
    print(f"Counts {'match' if image_count == label_count else 'do not match'}!")
    
    return image_count == label_count

# *****************************************************************************
# 메인 실행 함수
def main():
    print("\n=== 데이터 정리 작업 시작 ===")
    
    # 1단계: train.txt 경로 확인
    train_txt_path = sys.argv[1] if len(sys.argv) > 1 else TRAIN_TXT_PATH
    if not os.path.exists(train_txt_path):
        print(f"Error: train.txt not found at {train_txt_path}")
        sys.exit(1)
    print(f"1. train.txt 경로 확인: {train_txt_path}")

    # 2단계: 가장 먼저 백업 수행
    print("\n2. 데이터 백업 시작...")
    backup_data(train_txt_path)
    print("백업 완료!")

    # 3단계: 작업 디렉토리 생성 및 빈 레이블 파일 검사
    print("\n3. 빈 레이블 파일 검사 시작...")
    os.makedirs(os.path.dirname(PATHS['empty_list']), exist_ok=True)
    empty_files = check_empty_txt_files(train_txt_path)
    save_empty_files_list(empty_files, PATHS['empty_list'])
    
    # 4단계: 빈 레이블 파일에 대응하는 이미지 삭제
    if empty_files:
        print("\n4. 불필요한 이미지 파일 정리 시작...")
        remove_empty_label_images(empty_files)
        
        # 5단계: 최종 검증
        print("\n5. 최종 검증 시작...")
        first_image = empty_files[0][0]
        image_dir = os.path.dirname(first_image)
        label_dir = image_dir.replace('images', 'labels')
        verify_image_label_count(image_dir, label_dir)
    
    print("\n=== 데이터 정리 작업 완료 ===")
    print(f"작업 결과가 {PATHS['empty_list']}에 저장되었습니다.")

if __name__ == "__main__":
    main()
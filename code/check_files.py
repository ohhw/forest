import os
from pathlib import Path

def check_files():
    train_txt = "/hdd/datasets/dod_data/jjb/v8/train.txt"
    missing_images = []
    missing_labels = []
    
    with open(train_txt, 'r') as f:
        image_paths = f.read().splitlines()
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            missing_images.append(img_path)
        
        # 라벨 파일 경로를 생성
        label_path = img_path.replace('/images/', '/labels/').replace('.bmp', '.txt')
        if not os.path.exists(label_path):
            missing_labels.append(label_path)
    
    print(f"Total images in train.txt: {len(image_paths)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing labels: {len(missing_labels)}")
    
    if len(missing_images) > 0:
        print("\nFirst 5 missing images:")
        for path in missing_images[:5]:
            print(path)
            
    if len(missing_labels) > 0:
        print("\nFirst 5 missing labels:")
        for path in missing_labels[:5]:
            print(path)

if __name__ == "__main__":
    check_files()

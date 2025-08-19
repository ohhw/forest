import os
import shutil

# 원본 라벨 파일 경로
labels_dir = r"/hdd/datasets/dod_data/jjb/old/v4/train/data/labels"
output_base_dir = r"/hdd/datasets/dod_data/jjb/v7/labels"

# 클래스별 디렉터리 생성
class_dirs = {str(i): os.path.join(output_base_dir, str(i)) for i in range(4)}

for dir_path in class_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# 라벨 파일 순회
def classify_labels():
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):  # 텍스트 파일만 처리
            file_path = os.path.join(labels_dir, filename)
            
            # 클래스 판별
            with open(file_path, "r") as f:
                lines = f.readlines()
                if not lines:
                    continue  # 빈 파일은 무시
                
                classes = {line.split()[0] for line in lines}  # 모든 바운딩 박스의 클래스 수집
                for class_id in classes:
                    if class_id in class_dirs:
                        shutil.copy(file_path, os.path.join(class_dirs[class_id], filename))

if __name__ == "__main__":
    classify_labels()
    print("파일 정리가 완료되었습니다.")

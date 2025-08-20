import os
import cv2
import yaml
from pathlib import Path


def check_dataset_images(yaml_path):
    """데이터셋의 모든 이미지 파일을 검증"""
    
    # YAML 파일 읽기
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 데이터셋 경로 확인
    dataset_path = data.get('path', '')
    train_txt = data.get('train', '')
    val_txt = data.get('val', '')
    
    print(f"Dataset path: {dataset_path}")
    print(f"Train txt: {train_txt}")
    print(f"Val txt: {val_txt}")
    
    corrupted_files = []
    
    # 각 데이터셋 경로 확인
    for split_name, txt_file in [('train', train_txt), ('val', val_txt)]:
        if not txt_file:
            continue
            
        print(f"\n=== {split_name.upper()} 데이터셋 검증 ===")
        
        # txt 파일 경로 구성
        if os.path.isabs(txt_file):
            txt_path = Path(txt_file)
        else:
            txt_path = Path(dataset_path) / txt_file
            
        if not txt_path.exists():
            print(f"txt 파일이 존재하지 않습니다: {txt_path}")
            continue
        
        # txt 파일에서 이미지 경로 목록 읽기
        with open(txt_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"총 {len(image_paths)}개 이미지 파일 발견")
        
        # 각 이미지 파일 검증
        for i, img_path_str in enumerate(image_paths):
            try:
                # 절대 경로로 변환
                if os.path.isabs(img_path_str):
                    img_path = Path(img_path_str)
                else:
                    img_path = Path(dataset_path) / img_path_str
                
                if not img_path.exists():
                    corrupted_files.append(str(img_path))
                    print(f"[ERROR] 파일이 존재하지 않음: {img_path}")
                    continue
                
                # OpenCV로 이미지 읽기 시도
                img = cv2.imread(str(img_path))
                
                if img is None:
                    corrupted_files.append(str(img_path))
                    print(f"[ERROR] 읽을 수 없는 이미지: {img_path}")
                    continue
                
                # 이미지 크기 확인
                h, w = img.shape[:2]
                if h == 0 or w == 0:
                    corrupted_files.append(str(img_path))
                    print(f"[ERROR] 잘못된 크기의 이미지: {img_path} (크기: {w}x{h})")
                    continue
                
                # 간단한 변환 테스트 (affine transform과 유사)
                try:
                    resized = cv2.resize(img, (640, 640))
                except Exception as e:
                    corrupted_files.append(str(img_path))
                    print(f"[ERROR] 리사이즈 실패: {img_path} - {e}")
                    continue
                    
            except Exception as e:
                corrupted_files.append(str(img_path_str))
                print(f"[ERROR] 이미지 처리 실패: {img_path_str} - {e}")
            
            # 진행상황 표시
            if (i + 1) % 100 == 0:
                print(f"진행중... {i + 1}/{len(image_paths)}")
    
    # 결과 출력
    print(f"\n=== 검증 완료 ===")
    print(f"손상된 이미지 파일 수: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\n손상된 파일 목록:")
        for file in corrupted_files:
            print(f"  - {file}")
        
        # 손상된 파일 목록을 텍스트 파일로 저장
        with open('corrupted_images.txt', 'w') as f:
            for file in corrupted_files:
                f.write(f"{file}\n")
        
        print(f"\n손상된 파일 목록이 'corrupted_images.txt'에 저장되었습니다.")
        print("이 파일들을 삭제하거나 교체한 후 학습을 다시 시도해보세요.")
    else:
        print("모든 이미지 파일이 정상입니다!")


if __name__ == "__main__":
    # 설정값
    Product = "jjb"
    data_nm = "v10"
    
    yaml_path = f"/hdd/datasets/dod_data/{Product}/{Product}_defect_detection_data_{data_nm}.yaml"
    
    print(f"데이터셋 YAML 파일: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        print(f"YAML 파일이 존재하지 않습니다: {yaml_path}")
        exit(1)
    
    check_dataset_images(yaml_path)

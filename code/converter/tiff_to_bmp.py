# tiff to bmp
from PIL import Image
import os

def convert_tiff_to_bmp_in_folders(base_dir, subfolder_count, output_dir):
    # 저장 폴더 확인
    if not os.path.exists(output_dir):
        print(f"저장 폴더가 존재하지 않습니다: {output_dir}")
        return
    
    for i in range(24, 24 + subfolder_count + 1):
        folder_path = os.path.join(base_dir, str(i))
        
        # 폴더가 존재하지 않을 경우 스킵
        if not os.path.exists(folder_path):
            print(f"폴더가 존재하지 않습니다: {folder_path}")
            continue
        
        print(f"탐색 중: {folder_path}")
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.tiff') or file_name.lower().endswith('.tif'):
                input_file = os.path.join(folder_path, file_name)
                output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.bmp")
                
                try:
                    with Image.open(input_file) as img:
                        img.save(output_file, "BMP")
                    print(f"변환 성공: {input_file} -> {output_file}")
                except Exception as e:
                    print(f"변환 실패: {input_file}, 오류: {e}")
    print("종료")

subfolder_count = 25  # 탐색할 하위 폴더 수

# wnl 탐색 경로
# base_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_wln\raw_tiff\호두_240926_1차_필터링\양품"
# output_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_wln\raw_bmp\good"  # 저장 경로

# csn 탐색 경로
base_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_csn\raw_tiff\good"
output_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_csn\raw_bmp\good"  # 저장 경로

# psm 탐색 경로
# base_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_psm\raw_tiff"
# output_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_psm\raw_bmp"  # 저장 경로

# # jjb 탐색 경로
# base_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_jjb\raw_tiff\good"
# output_directory = r"C:\Users\USER\Desktop\data\2024_defect_detection\raw_data\raw_jjb\raw_bmp\good\241224"  # 저장 경로

convert_tiff_to_bmp_in_folders(base_directory, subfolder_count, output_directory)

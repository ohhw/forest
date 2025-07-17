import os
import shutil


def split_images_into_subfolders(source_dir, files_per_folder=150):
    """
    이미지 파일을 지정된 개수만큼 하위 폴더에 나누어 저장

    Args:
        source_dir: 이미지가 있는 폴더 경로
        files_per_folder: 각 하위 폴더에 저장할 파일 개수
    """
    # 이미지 파일 확장자 목록
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]

    # 디렉토리 내 모든 이미지 파일 목록 가져오기
    all_files = [
        f
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f))
        and os.path.splitext(f)[1].lower() in image_extensions
    ]

    print(f"총 {len(all_files)}개의 이미지 파일을 발견했습니다.")

    # 필요한 하위 폴더 개수 계산
    total_files = len(all_files)
    num_folders = (total_files + files_per_folder - 1) // files_per_folder

    print(
        f"각 폴더당 {files_per_folder}개씩, 총 {num_folders}개의 하위 폴더가 생성됩니다."
    )

    # 하위 폴더 생성 및 파일 이동
    for folder_idx in range(1, num_folders + 1):
        # 하위 폴더 경로 (숫자로 폴더명 지정)
        subfolder_path = os.path.join(source_dir, str(folder_idx))

        # 하위 폴더가 없으면 생성
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # 이 하위 폴더에 넣을 파일 범위 계산
        start_idx = (folder_idx - 1) * files_per_folder
        end_idx = min(folder_idx * files_per_folder, total_files)

        # 파일 이동
        for i in range(start_idx, end_idx):
            src_file = os.path.join(source_dir, all_files[i])
            dst_file = os.path.join(subfolder_path, all_files[i])
            shutil.move(src_file, dst_file)

        print(f"폴더 {folder_idx}에 {end_idx - start_idx}개 파일 이동 완료")


# 사용 예시
split_images_into_subfolders("C:/Users/oh/Desktop/forest/jjb/SJ1")

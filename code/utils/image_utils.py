"""
이미지 파일 관련 유틸리티 함수
"""

import os
from glob import glob
from pathlib import Path
from typing import Union, List, Set

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {
    '.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.gif',
    '.BMP', '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF', '.WEBP', '.GIF'
}


def is_image_file(file_path: Union[str, Path]) -> bool:
    """
    파일이 이미지 파일인지 확인합니다.
    
    Args:
        file_path: 확인할 파일 경로
        
    Returns:
        이미지 파일 여부
    """
    ext = os.path.splitext(str(file_path))[1]
    return ext in IMAGE_EXTENSIONS


def collect_images(directory: Union[str, Path], 
                   extensions: Set[str] = None,
                   recursive: bool = False) -> List[str]:
    """
    디렉토리에서 이미지 파일을 수집합니다.
    
    Args:
        directory: 검색할 디렉토리
        extensions: 검색할 확장자 집합 (None이면 기본값 사용)
        recursive: True이면 하위 디렉토리까지 재귀 검색
        
    Returns:
        이미지 파일 경로 리스트 (절대 경로)
    """
    if not os.path.exists(directory):
        print(f"⚠️  디렉토리가 존재하지 않습니다: {directory}")
        return []
    
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    
    image_list = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        # 점(.)이 없으면 추가
        if not ext.startswith('.'):
            ext = '.' + ext
        
        search_pattern = os.path.join(directory, f"{pattern}{ext}")
        found = glob(search_pattern, recursive=recursive)
        image_list.extend(found)
    
    # 절대 경로로 변환 및 정렬
    image_list = [os.path.abspath(path) for path in image_list]
    image_list.sort()
    
    return image_list


def get_image_count(directory: Union[str, Path], recursive: bool = False) -> int:
    """
    디렉토리의 이미지 파일 개수를 반환합니다.
    
    Args:
        directory: 검색할 디렉토리
        recursive: True이면 하위 디렉토리까지 재귀 검색
        
    Returns:
        이미지 파일 개수
    """
    images = collect_images(directory, recursive=recursive)
    return len(images)


def filter_image_files(file_list: List[Union[str, Path]]) -> List[str]:
    """
    파일 리스트에서 이미지 파일만 필터링합니다.
    
    Args:
        file_list: 필터링할 파일 경로 리스트
        
    Returns:
        이미지 파일만 포함된 리스트
    """
    return [str(f) for f in file_list if is_image_file(f)]


def get_image_extension_stats(directory: Union[str, Path], 
                              recursive: bool = False) -> dict:
    """
    디렉토리의 이미지 파일 확장자별 통계를 반환합니다.
    
    Args:
        directory: 검색할 디렉토리
        recursive: True이면 하위 디렉토리까지 재귀 검색
        
    Returns:
        {확장자: 개수} 딕셔너리
        
    Examples:
        >>> get_image_extension_stats("/data/images")
        {'.jpg': 150, '.png': 50, '.bmp': 20}
    """
    images = collect_images(directory, recursive=recursive)
    
    stats = {}
    for img_path in images:
        ext = os.path.splitext(img_path)[1].lower()
        stats[ext] = stats.get(ext, 0) + 1
    
    return stats


def check_image_readable(image_path: Union[str, Path]) -> bool:
    """
    이미지 파일이 읽을 수 있는지 확인합니다.
    
    Args:
        image_path: 확인할 이미지 경로
        
    Returns:
        읽기 가능 여부
    """
    try:
        import cv2
        img = cv2.imread(str(image_path))
        return img is not None
    except Exception:
        return False

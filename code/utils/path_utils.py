"""
경로 관련 유틸리티 함수
"""

import os
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path], create: bool = True) -> str:
    """
    디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
    
    Args:
        path: 확인할 디렉토리 경로
        create: True이면 디렉토리가 없을 때 생성
        
    Returns:
        절대 경로 문자열
        
    Raises:
        FileNotFoundError: 디렉토리가 없고 create=False인 경우
    """
    path = get_absolute_path(path)
    
    if not os.path.exists(path):
        if create:
            os.makedirs(path, exist_ok=True)
        else:
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {path}")
    
    return path


def get_absolute_path(path: Union[str, Path]) -> str:
    """
    상대 경로를 절대 경로로 변환합니다.
    
    Args:
        path: 변환할 경로
        
    Returns:
        절대 경로 문자열
        
    Examples:
        >>> get_absolute_path("./data")
        '/home/user/project/data'
        >>> get_absolute_path("~/data")
        '/home/user/data'
    """
    path = os.path.expanduser(str(path))  # ~ 확장
    path = os.path.abspath(path)          # 절대 경로 변환
    return os.path.normpath(path)         # 경로 정규화


def check_path_exists(path: Union[str, Path], path_type: str = "any") -> bool:
    """
    경로가 존재하는지 확인합니다.
    
    Args:
        path: 확인할 경로
        path_type: 'file', 'dir', 'any' 중 하나
        
    Returns:
        존재 여부
    """
    path = str(path)
    
    if not os.path.exists(path):
        return False
    
    if path_type == "file":
        return os.path.isfile(path)
    elif path_type == "dir":
        return os.path.isdir(path)
    else:  # any
        return True


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
    """
    base를 기준으로 한 상대 경로를 반환합니다.
    
    Args:
        path: 대상 경로
        base: 기준 경로
        
    Returns:
        상대 경로 문자열
    """
    path = get_absolute_path(path)
    base = get_absolute_path(base)
    return os.path.relpath(path, base)


def split_path(path: Union[str, Path]) -> tuple:
    """
    경로를 디렉토리, 파일명, 확장자로 분리합니다.
    
    Args:
        path: 분리할 경로
        
    Returns:
        (디렉토리, 파일명, 확장자) 튜플
        
    Examples:
        >>> split_path("/data/image.jpg")
        ('/data', 'image', '.jpg')
    """
    path = str(path)
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    return directory, name, ext

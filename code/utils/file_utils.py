"""
파일 입출력 관련 유틸리티 함수
"""

import os
import shutil
from pathlib import Path
from typing import Union, List
from .path_utils import ensure_dir, get_absolute_path
from .time_utils import format_timestamp


def read_file_lines(file_path: Union[str, Path], strip: bool = True, skip_empty: bool = False) -> List[str]:
    """
    파일을 줄 단위로 읽어서 리스트로 반환합니다.
    
    Args:
        file_path: 읽을 파일 경로
        strip: True이면 각 줄의 앞뒤 공백 제거
        skip_empty: True이면 빈 줄 제외
        
    Returns:
        줄 단위 리스트
    """
    file_path = get_absolute_path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if strip:
        lines = [line.strip() for line in lines]
    
    if skip_empty:
        lines = [line for line in lines if line]
    
    return lines


def write_file_lines(file_path: Union[str, Path], lines: List[str], 
                     create_dir: bool = True, append: bool = False) -> bool:
    """
    리스트를 파일에 줄 단위로 씁니다.
    
    Args:
        file_path: 쓸 파일 경로
        lines: 쓸 내용 리스트
        create_dir: True이면 디렉토리 자동 생성
        append: True이면 파일 끝에 추가
        
    Returns:
        성공 여부
    """
    try:
        file_path = get_absolute_path(file_path)
        
        if create_dir:
            directory = os.path.dirname(file_path)
            if directory:
                ensure_dir(directory)
        
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as f:
            for line in lines:
                if not line.endswith('\n'):
                    line += '\n'
                f.write(line)
        
        return True
    except Exception as e:
        print(f"파일 쓰기 실패: {e}")
        return False


def backup_file(file_path: Union[str, Path], backup_dir: str = None, 
                add_timestamp: bool = True) -> str:
    """
    파일을 백업합니다.
    
    Args:
        file_path: 백업할 파일 경로
        backup_dir: 백업 디렉토리 (None이면 원본과 같은 위치)
        add_timestamp: True이면 파일명에 타임스탬프 추가
        
    Returns:
        백업된 파일 경로
    """
    file_path = get_absolute_path(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"백업할 파일을 찾을 수 없습니다: {file_path}")
    
    # 백업 디렉토리 설정
    if backup_dir is None:
        backup_dir = os.path.dirname(file_path)
    else:
        backup_dir = get_absolute_path(backup_dir)
        ensure_dir(backup_dir)
    
    # 백업 파일명 생성
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    
    if add_timestamp:
        timestamp = format_timestamp()
        backup_filename = f"{name}_{timestamp}{ext}"
    else:
        backup_filename = f"{name}_backup{ext}"
    
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # 백업 수행
    shutil.copy2(file_path, backup_path)
    
    return backup_path


def copy_file(src: Union[str, Path], dst: Union[str, Path], 
              create_dir: bool = True, overwrite: bool = False) -> bool:
    """
    파일을 복사합니다.
    
    Args:
        src: 원본 파일 경로
        dst: 대상 파일 경로
        create_dir: True이면 대상 디렉토리 자동 생성
        overwrite: True이면 기존 파일 덮어쓰기
        
    Returns:
        성공 여부
    """
    try:
        src = get_absolute_path(src)
        dst = get_absolute_path(dst)
        
        if not os.path.exists(src):
            print(f"원본 파일을 찾을 수 없습니다: {src}")
            return False
        
        if os.path.exists(dst) and not overwrite:
            print(f"대상 파일이 이미 존재합니다: {dst}")
            return False
        
        if create_dir:
            dst_dir = os.path.dirname(dst)
            if dst_dir:
                ensure_dir(dst_dir)
        
        shutil.copy2(src, dst)
        return True
        
    except Exception as e:
        print(f"파일 복사 실패: {e}")
        return False


def move_file(src: Union[str, Path], dst: Union[str, Path], 
              create_dir: bool = True, overwrite: bool = False) -> bool:
    """
    파일을 이동합니다.
    
    Args:
        src: 원본 파일 경로
        dst: 대상 파일 경로
        create_dir: True이면 대상 디렉토리 자동 생성
        overwrite: True이면 기존 파일 덮어쓰기
        
    Returns:
        성공 여부
    """
    try:
        src = get_absolute_path(src)
        dst = get_absolute_path(dst)
        
        if not os.path.exists(src):
            print(f"원본 파일을 찾을 수 없습니다: {src}")
            return False
        
        if os.path.exists(dst) and not overwrite:
            print(f"대상 파일이 이미 존재합니다: {dst}")
            return False
        
        if create_dir:
            dst_dir = os.path.dirname(dst)
            if dst_dir:
                ensure_dir(dst_dir)
        
        shutil.move(src, dst)
        return True
        
    except Exception as e:
        print(f"파일 이동 실패: {e}")
        return False

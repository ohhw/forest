"""
유틸리티 함수 모음
"""

from pathlib import Path
from typing import Optional


def ensure_dir(path: str) -> Path:
    """
    디렉토리가 없으면 생성
    
    Args:
        path: 디렉토리 경로
        
    Returns:
        Path 객체
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_weights(runs_dir: str, model_prefix: Optional[str] = None) -> Optional[str]:
    """
    runs 디렉토리에서 가장 최근 best.pt 찾기
    
    Args:
        runs_dir: runs 디렉토리 경로
        model_prefix: 모델 이름 prefix (예: 'csn_dod')
        
    Returns:
        best.pt 경로 (없으면 None)
    """
    runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        return None
    
    # 모든 best.pt 찾기
    candidates = []
    for weight_path in runs_dir.rglob("weights/best.pt"):
        if model_prefix is None or model_prefix in str(weight_path):
            candidates.append(weight_path)
    
    if not candidates:
        return None
    
    # 수정 시간 기준으로 정렬하여 최신 것 반환
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def format_time(seconds: float) -> str:
    """
    초를 시:분:초 형식으로 변환
    
    Args:
        seconds: 초
        
    Returns:
        "HH:MM:SS" 형식 문자열
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

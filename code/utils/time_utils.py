"""
시간 관련 유틸리티 함수
"""

from datetime import datetime
import pytz


def format_time(seconds: float) -> str:
    """
    초를 시:분:초 형식으로 변환합니다.
    
    Args:
        seconds: 초 단위 시간
        
    Returns:
        "X시간 Y분 Z초" 형식의 문자열
        
    Examples:
        >>> format_time(3665)
        '1시간 1분 5초'
        >>> format_time(65)
        '0시간 1분 5초'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}시간 {minutes}분 {secs}초"


def format_timestamp(format_str: str = "%Y%m%d_%H%M%S", timezone: str = "Asia/Seoul") -> str:
    """
    현재 시간을 지정된 형식의 문자열로 반환합니다.
    
    Args:
        format_str: strftime 형식 문자열
        timezone: 타임존 (기본: Asia/Seoul)
        
    Returns:
        포맷팅된 시간 문자열
        
    Examples:
        >>> format_timestamp()
        '20250101_153045'
        >>> format_timestamp("%Y-%m-%d %H:%M:%S")
        '2025-01-01 15:30:45'
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return now.strftime(format_str)


def get_elapsed_time_str(start_time: float, end_time: float = None) -> str:
    """
    시작 시간과 종료 시간 사이의 경과 시간을 문자열로 반환합니다.
    
    Args:
        start_time: 시작 시간 (time.time())
        end_time: 종료 시간 (None이면 현재 시간)
        
    Returns:
        경과 시간 문자열
    """
    import time
    if end_time is None:
        end_time = time.time()
    
    elapsed = end_time - start_time
    return format_time(elapsed)

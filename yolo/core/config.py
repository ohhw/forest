"""
설정 파일 로더
YAML 설정 파일을 로드하고 base.yaml과 병합
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigLoader:
    """YAML 설정 파일을 로드하고 병합하는 클래스"""
    
    def __init__(self, config_path: str, task: Optional[str] = None):
        """
        Args:
            config_path: 임산물 설정 파일 경로 (예: configs/models/dod/csn.yaml)
            task: 'detect' or 'classify' (None이면 자동 감지)
        """
        self.config_path = Path(config_path)
        self.task = task
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    def load(self) -> Dict[str, Any]:
        """
        base.yaml과 임산물 설정을 병합하여 반환
        
        Returns:
            병합된 설정 딕셔너리
        """
        # 1. base.yaml 로드
        base_path = self.config_path.parent / "base.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"base.yaml을 찾을 수 없습니다: {base_path}")
        
        with open(base_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 2. 임산물별 설정 로드
        with open(self.config_path, 'r', encoding='utf-8') as f:
            product_config = yaml.safe_load(f)
        
        # 3. 병합 (product_config가 base_config를 덮어씀)
        config = self._deep_merge(base_config, product_config)
        
        # 4. task 자동 감지 또는 설정
        if self.task is not None:
            config['task'] = self.task
        elif 'task' not in config:
            # 경로에서 task 추론 (dod -> detect, cls -> classify)
            if 'dod' in str(self.config_path):
                config['task'] = 'detect'
            elif 'cls' in str(self.config_path):
                config['task'] = 'classify'
            else:
                raise ValueError("task를 명시하거나 설정 파일에 포함시켜주세요")
        
        return config
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        딕셔너리 깊은 병합 (재귀적으로 병합)
        
        Args:
            base: 기본 설정 딕셔너리
            override: 덮어쓸 설정 딕셔너리
            
        Returns:
            병합된 딕셔너리
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 둘 다 딕셔너리면 재귀적으로 병합
                result[key] = self._deep_merge(result[key], value)
            else:
                # 그 외에는 덮어쓰기
                result[key] = value
        
        return result
    
    def save(self, config: Dict[str, Any], output_path: str):
        """
        설정을 YAML 파일로 저장
        
        Args:
            config: 저장할 설정 딕셔너리
            output_path: 저장할 파일 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"[INFO] 설정 저장 완료: {output_path}")
    
    @staticmethod
    def load_yaml(yaml_path: str) -> Dict[str, Any]:
        """
        단순 YAML 파일 로드 (병합 없이)
        
        Args:
            yaml_path: YAML 파일 경로
            
        Returns:
            YAML 내용 딕셔너리
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

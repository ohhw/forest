"""
YOLO 모델 추론 관리자
학습된 모델로 이미지 추론 실행
"""

from ultralytics import YOLO
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


class YOLOPredictor:
    """YOLO 모델 추론을 담당하는 클래스"""
    
    def __init__(self, config: Dict[str, Any], weights_path: str):
        """
        Args:
            config: 설정 딕셔너리 (ConfigLoader.load() 결과)
            weights_path: 학습된 모델 가중치 경로 (.pt 파일)
        """
        self.config = config
        self.weights_path = Path(weights_path)
        self.product = config['product']
        self.task = config.get('task', 'detect')
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")
        
        # 모델 로드
        print(f"[INFO] 모델 로드: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        
    def predict(self,
                source: Union[str, List[str]],
                save: Optional[bool] = None,
                save_crop: Optional[bool] = None,
                save_txt: Optional[bool] = None,
                show_conf: bool = True,
                conf: Optional[float] = None,
                iou: Optional[float] = None,
                output_name: Optional[str] = None) -> List:
        """
        추론 실행
        
        Args:
            source: 추론할 이미지 경로 (파일, 디렉토리, 또는 리스트)
            save: 결과 이미지 저장 여부 (None이면 설정 파일 사용)
            save_crop: crop 이미지 저장 여부 (detection only)
            save_txt: 라벨 txt 저장 여부
            show_conf: confidence 표시 여부
            conf: confidence threshold (None이면 설정 파일 사용)
            iou: IoU threshold (None이면 설정 파일 사용)
            output_name: 출력 폴더 이름 (None이면 자동 생성)
            
        Returns:
            추론 결과 리스트
        """
        pred_config = self.config.get('prediction', {})
        
        # 설정 병합 (인자가 우선, 없으면 설정 파일, 그것도 없으면 기본값)
        save = save if save is not None else pred_config.get('save', True)
        save_txt = save_txt if save_txt is not None else pred_config.get('save_txt', True)
        conf_threshold = conf if conf is not None else pred_config.get('conf')
        iou_threshold = iou if iou is not None else pred_config.get('iou')
        
        # Detection의 경우만 save_crop 사용
        if self.task == 'detect':
            save_crop = save_crop if save_crop is not None else pred_config.get('save_crop', True)
        else:
            save_crop = False
        
        # 출력 이름 생성
        if output_name is None:
            output_name = self._generate_pred_name(show_conf)
        
        task_emoji = "🎯" if self.task == 'detect' else "🏷️"
        task_name = "Detection" if self.task == 'detect' else "Classification"
        
        print(f"[INFO] ==========================================")
        print(f"[INFO] {task_emoji} {task_name} 추론 시작")
        print(f"[INFO] - Task: {task_name}")
        print(f"[INFO] - Product: {self.product.upper()}")
        print(f"[INFO] - Source: {source}")
        print(f"[INFO] - Confidence: {conf_threshold}")
        print(f"[INFO] - Show conf: {show_conf}")
        if self.task == 'detect':
            print(f"[INFO] - Save crop: {save_crop}")
        print(f"[INFO] ==========================================")
        
        # 추론 파라미터 준비
        predict_params = {
            'source': source,
            'save': save,
            'save_txt': save_txt,
            'show_conf': show_conf,
            'exist_ok': True,
            'project': self._get_output_dir(),
            'name': output_name,
        }
        
        # 선택적 파라미터 추가
        if conf_threshold is not None:
            predict_params['conf'] = conf_threshold
        if iou_threshold is not None:
            predict_params['iou'] = iou_threshold
        if save_crop and self.task == 'detect':
            predict_params['save_crop'] = save_crop
        
        # 추론 실행
        results = self.model.predict(**predict_params)
        
        task_emoji = "🎯" if self.task == 'detect' else "🏷️"
        task_name = "Detection" if self.task == 'detect' else "Classification"
        
        print(f"[INFO] {task_emoji} {task_name} 추론 완료: {len(results)}개 이미지")
        print(f"[INFO] 결과 저장: {self._get_output_dir()}/{output_name}")
        
        return results
    
    def predict_with_without_conf(self,
                                   source: Union[str, List[str]],
                                   conf: Optional[float] = None) -> tuple:
        """
        confidence 표시 on/off 두 버전 모두 실행
        기존 코드의 패턴을 재현
        
        Args:
            source: 추론할 이미지 경로
            conf: confidence threshold
            
        Returns:
            (results_with_conf, results_without_conf) 튜플
        """
        task_emoji = "🎯" if self.task == 'detect' else "🏷️"
        task_name = "Detection" if self.task == 'detect' else "Classification"
        
        print(f"[INFO] {task_emoji} {task_name} - Confidence 표시 ON/OFF 두 버전 추론 시작")
        
        # 1. Confidence 표시 O
        results_with = self.predict(
            source=source,
            show_conf=True,
            conf=conf,
        )
        
        # 2. Confidence 표시 X
        results_without = self.predict(
            source=source,
            show_conf=False,
            conf=conf,
        )
        
        return results_with, results_without
    
    def _generate_pred_name(self, show_conf: bool) -> str:
        """
        추론 결과 폴더명 생성
        
        Args:
            show_conf: confidence 표시 여부
            
        Returns:
            생성된 폴더 이름
        """
        # 가중치 파일에서 모델 이름 추출 시도
        weight_name = self.weights_path.stem  # 확장자 제거
        
        # best 또는 last 제거
        if weight_name.endswith('_best') or weight_name.endswith('_last'):
            weight_name = weight_name.rsplit('_', 1)[0]
        
        # Confidence 유무에 따른 suffix
        suffix = "" if show_conf else "_without_conf"
        
        # 날짜 추가
        date = datetime.now().strftime('%y%m%d')
        
        return f"pred_{weight_name}_{date}{suffix}"
    
    def _get_output_dir(self) -> str:
        """
        추론 결과 저장 디렉토리
        
        Returns:
            runs 디렉토리 경로
        """
        paths = self.config['paths']
        product = self.config['product']
        
        return f"{paths['output_root']}/{product}/runs"
    
    def get_validation_path(self) -> str:
        """
        검증 데이터셋 경로 반환 (편의 메서드)
        
        Returns:
            검증 이미지 디렉토리 경로
        """
        paths = self.config['paths']
        product = self.config['product']
        data_version = self.config.get('data_version', 'v2')
        
        if self.task == 'detect':
            # Detection: val2/images 또는 val/images
            base_path = Path(paths['data_root']) / product
            
            # val2가 있으면 val2 사용 (우선순위)
            val2_path = base_path / "val2" / "images"
            if val2_path.exists():
                return str(val2_path)
            
            # 없으면 val 사용
            val_path = base_path / data_version / "val" / "images"
            if val_path.exists():
                return str(val_path)
            
            # 그것도 없으면 기본 val
            return str(base_path / "val" / "images")
        else:
            # Classification: val 디렉토리
            return f"{paths['data_root']}/{product}/{data_version}/val"

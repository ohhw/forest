"""
YOLO 모델 학습 관리자
학습 로직을 캡슐화하고 경로 및 이름 생성 자동화
"""

from ultralytics import YOLO
import os
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Classification 평가를 위한 import (optional)
try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class YOLOTrainer:
    """YOLO 모델 학습을 담당하는 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리 (ConfigLoader.load() 결과)
        """
        self.config = config
        self.product = config['product']
        self.task = config.get('task', 'detect')
        self.model_name = None  # 학습 시 생성됨
        
        # 작업 디렉토리 설정
        self._setup_working_directory()
    
    def _setup_working_directory(self):
        """작업 디렉토리 및 torch hub 경로 설정"""
        paths = self.config['paths']
        product_dir = Path(paths['output_root']) / self.product
        product_dir.mkdir(parents=True, exist_ok=True)
        
        # 작업 디렉토리 변경
        os.chdir(product_dir)
        
        # Torch Hub 캐시 경로 설정
        torch.hub.set_dir(str(product_dir))
        
        print(f"[INFO] 작업 디렉토리: {product_dir}")
    
    def setup_model(self) -> YOLO:
        """
        사전학습 모델 로드
        
        Returns:
            YOLO 모델 객체
        """
        model_name = self.config['model']
        model_path = self._get_pretrained_model_path(model_name)
        
        if not Path(model_path).exists():
            print(f"[WARNING] 사전학습 모델이 없습니다: {model_path}")
            print(f"[INFO] Ultralytics에서 자동으로 다운로드합니다...")
        
        print(f"[INFO] 모델 로드: {model_path}")
        return YOLO(model_path)
    
    def _get_pretrained_model_path(self, model_name: str) -> str:
        """
        사전학습 모델 경로 반환
        
        Args:
            model_name: 모델 이름 (예: yolo11n, yolo11s-cls)
            
        Returns:
            모델 파일 전체 경로
        """
        # 1. 설정 파일에 pretrained_models 경로가 있으면 사용
        if 'pretrained_models' in self.config['paths']:
            pretrained_dir = Path(self.config['paths']['pretrained_models'])
            model_path = pretrained_dir / f"{model_name}.pt"
            if model_path.exists():
                return str(model_path)
        
        # 2. 현재 임산물 디렉토리에 있는지 확인
        product_dir = Path(self.config['paths']['output_root']) / self.product
        model_path = product_dir / f"{model_name}.pt"
        if model_path.exists():
            return str(model_path)
        
        # 3. 없으면 모델 이름만 반환 (Ultralytics가 자동 다운로드)
        return f"{model_name}.pt"
    
    def train(self, model: YOLO, resume: bool = False) -> str:
        """
        학습 실행
        
        Args:
            model: YOLO 모델 객체
            resume: 이전 학습 재개 여부
            
        Returns:
            학습된 모델의 best.pt 경로
        """
        # 모델 이름 생성 (csn_dod_11n_25121910h)
        self.model_name = self._generate_model_name()
        
        # 설정에서 학습 파라미터 추출
        train_config = self.config['training']
        
        # 데이터 경로 구성
        data_path = self._get_data_path()
        
        print(f"[INFO] ==========================================")
        print(f"[INFO] 학습 시작")
        print(f"[INFO] - 모델명: {self.model_name}")
        print(f"[INFO] - 데이터: {data_path}")
        print(f"[INFO] - Task: {self.task}")
        print(f"[INFO] ==========================================")
        
        start_time = time.time()
        
        # 학습 파라미터 준비
        train_params = {
            'data': data_path,
            'epochs': train_config.get('epochs', 100),
            'batch': train_config.get('batch', 32),
            'patience': train_config.get('patience', 50),
            'exist_ok': self.config['common'].get('exist_ok', True),
            'project': self._get_output_dir(),
            'name': self.model_name,
        }
        
        # 선택적 파라미터 추가
        optional_params = ['dropout', 'iou', 'lr0', 'lrf', 'optimizer', 'imgsz', 'plots']
        for param in optional_params:
            if param in train_config:
                train_params[param] = train_config[param]
        
        # resume 옵션 추가
        if resume:
            train_params['resume'] = True
        
        # 학습 실행
        try:
            model.train(**train_params)
        except Exception as e:
            print(f"[ERROR] 학습 중 오류 발생: {e}")
            raise
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print(f"[INFO] ==========================================")
        print(f"[INFO] 학습 완료!")
        print(f"[INFO] - 소요 시간: {hours}시간 {minutes}분 {seconds}초")
        print(f"[INFO] ==========================================")
        
        # Tensorboard 안내
        print(f"[INFO] 텐서보드 실행: tensorboard --logdir {self._get_output_dir()}")
        
        # best.pt 경로 반환
        best_path = self._get_best_weight_path(self.model_name)
        print(f"[INFO] Best weights: {best_path}")
        
        return best_path
    
    def validate(self, model: YOLO, split: str = 'val') -> Dict[str, Any]:
        """
        검증 실행
        
        Args:
            model: YOLO 모델 객체
            split: 검증 데이터셋 분할 ('val', 'test')
            
        Returns:
            검증 메트릭 딕셔너리
        """
        print(f"[INFO] 검증 시작 (split: {split})")
        
        data_path = self._get_data_path()
        metrics = model.val(data=data_path, split=split)
        
        print(f"[INFO] 검증 완료")
        return metrics
    
    def _get_data_path(self) -> str:
        """
        데이터셋 경로 생성
        
        Returns:
            데이터셋 YAML 경로 (detection) 또는 디렉토리 경로 (classification)
        """
        paths = self.config['paths']
        product = self.config['product']
        data_version = self.config.get('data_version', 'v2')
        
        if self.task == 'detect':
            # Detection: YAML 파일 경로
            yaml_name = f"{product}_defect_detection_data_{data_version}.yaml"
            return f"{paths['data_root']}/{product}/{yaml_name}"
        else:
            # Classification: 디렉토리 경로
            return f"{paths['data_root']}/{product}/{data_version}"
    
    def _generate_model_name(self) -> str:
        """
        모델 이름 자동 생성
        형식: {product}_{task}_{model}_{date}{time}
        예: csn_dod_11n_25121910h
        
        Returns:
            생성된 모델 이름
        """
        now = datetime.now()
        date = now.strftime('%y%m%d')
        hour = now.strftime('%Hh')
        
        product = self.config['product']
        model = self.config['model'].replace('yolo', '').replace('.pt', '')
        task_prefix = 'dod' if self.task == 'detect' else 'cls'
        
        return f"{product}_{task_prefix}_{model}_{date}{hour}"
    
    def _get_output_dir(self) -> str:
        """
        출력 디렉토리 경로
        
        Returns:
            runs 디렉토리 경로
        """
        paths = self.config['paths']
        product = self.config['product']
        
        return f"{paths['output_root']}/{product}/runs"
    
    def _get_best_weight_path(self, model_name: str) -> str:
        """
        best.pt 경로 반환 (없으면 last.pt)
        
        Args:
            model_name: 모델 이름
            
        Returns:
            가중치 파일 경로
            
        Raises:
            FileNotFoundError: 가중치 파일을 찾을 수 없는 경우
        """
        output_dir = self._get_output_dir()
        best_path = Path(output_dir) / model_name / "weights" / "best.pt"
        
        if best_path.exists():
            return str(best_path)
        
        last_path = Path(output_dir) / model_name / "weights" / "last.pt"
        if last_path.exists():
            print(f"[WARNING] best.pt가 없어 last.pt를 사용합니다")
            return str(last_path)
        
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {best_path}")
    
    def load_trained_model(self, weights_path: str) -> YOLO:
        """
        학습된 모델 로드
        
        Args:
            weights_path: 가중치 파일 경로
            
        Returns:
            로드된 YOLO 모델
        """
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")
        
        return YOLO(weights_path)
    
    def predict_on_validation(self, model: YOLO, show_conf: bool = True) -> None:
        """
        검증 데이터셋으로 예측 실행 (기존 코드 호환)
        
        Args:
            model: YOLO 모델 객체
            show_conf: 예측 결과에 confidence 표시 여부 (Detection만 사용)
        
        Note:
            - Detection: 신뢰도 포함/제외 2번 실행 (show_conf 사용)
            - Classification: 1번만 실행 (show_conf 무시)
        """
        # 예측 설정 가져오기
        pred_config = self.config.get('prediction', {})
        
        # Detection의 경우 valid.txt 사용
        if self.task == 'detect':
            valid_txt_path = self._get_validation_images_path()
            
            if not Path(valid_txt_path).exists():
                print(f"[WARNING] {valid_txt_path} 파일이 없어 예측을 건너뜁니다.")
                return
            
            # valid.txt에서 이미지 경로 읽기
            with open(valid_txt_path, 'r') as f:
                valid_images = [line.strip() for line in f if line.strip()]
            
            print(f"[INFO] 검증 이미지 {len(valid_images)}개로 예측 실행 중...")
            source = valid_images
        else:
            # Classification의 경우 val 디렉토리 사용
            data_root = self.config['paths']['data_root']
            product = self.config['product']
            data_version = self.config.get('data_version', 'v2')
            val_dir = f"{data_root}/{product}/{data_version}/val"
            
            if not Path(val_dir).exists():
                print(f"[WARNING] {val_dir} 디렉토리가 없어 예측을 건너뜁니다.")
                return
            
            print(f"[INFO] {val_dir} 데이터로 예측 실행 중...")
            source = val_dir
        
        # 예측 파라미터 준비
        predict_params = {
            'source': source,
            'save': True,
            'save_txt': True,
            'exist_ok': True,
            'project': self._get_output_dir(),
        }
        
        # Detection 전용 파라미터
        if self.task == 'detect':
            predict_params.update({
                'save_crop': True,
                'show_conf': show_conf,
                'conf': pred_config.get('conf', 0.5),
            })
            
            # 선택적 파라미터
            optional_params = ['cls', 'kobj', 'dfl']
            for param in optional_params:
                if param in pred_config:
                    predict_params[param] = pred_config[param]
        
        # 예측 이름 설정
        conf_suffix = "" if show_conf else "_without_conf"
        predict_params['name'] = f"pred_{self.model_name}_val{conf_suffix}"
        
        # 예측 실행
        try:
            results = model.predict(**predict_params)
            
            # Classification의 경우 Confusion Matrix & Classification Report 자동 생성
            if self.task == 'classify' and SKLEARN_AVAILABLE and not show_conf:
                self._print_classification_metrics(results, source)
        except Exception as e:
            print(f"[ERROR] 예측 중 오류 발생: {e}")
    
    def _print_classification_metrics(self, results, val_dir):
        """
        Classification 예측 결과로 Confusion Matrix & Classification Report 출력
        
        Args:
            results: YOLO predict 결과
            val_dir: 검증 데이터 디렉토리 (str or list)
        """
        try:
            import numpy as np
            
            # val_dir이 리스트가 아니면 디렉토리에서 데이터 수집
            if isinstance(val_dir, str):
                val_path = Path(val_dir)
                if not val_path.exists():
                    return
                
                # 클래스 이름 수집
                class_names = sorted([d.name for d in val_path.iterdir() if d.is_dir()])
                
                # y_true 수집
                y_true = []
                img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.ppm')
                for class_idx, class_name in enumerate(class_names):
                    class_dir = val_path / class_name
                    for img_file in sorted(class_dir.iterdir()):
                        if img_file.suffix.lower() in img_exts:
                            y_true.append(class_idx)
            else:
                # 이미지 경로 리스트인 경우 (현재는 사용 안 함)
                return
            
            # y_pred 추출
            y_pred = []
            for result in results:
                if hasattr(result, 'probs'):
                    y_pred.append(result.probs.top1)
                else:
                    return  # Classification이 아니면 종료
            
            # 길이 체크
            if len(y_true) != len(y_pred):
                print(f"[WARNING] y_true({len(y_true)})와 y_pred({len(y_pred)}) 길이 불일치")
                return
            
            # Confusion Matrix 생성
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification Report 생성
            report = classification_report(y_true, y_pred, target_names=class_names)
            
            # 출력 (기존 코드와 동일한 형식)
            print(f"Confusion Matrix for YOLO :")
            print(cm)
            print(f"Classification Report for YOLO : ")
            print(report)
            
        except Exception as e:
            print(f"[WARNING] Classification 메트릭 생성 중 오류: {e}")
    
    def _get_validation_images_path(self) -> str:
        """
        검증 이미지 경로 텍스트 파일 경로 반환 (Detection용)
        
        Returns:
            valid.txt 파일 경로
        """
        paths = self.config['paths']
        product = self.config['product']
        data_version = self.config.get('data_version', 'v2')
        
        return f"{paths['data_root']}/{product}/{data_version}/valid.txt"

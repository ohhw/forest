"""
Classification 모델 평가
Confusion Matrix, Classification Report 생성
"""

import numpy as np
from pathlib import Path
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

try:
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] sklearn, matplotlib, seaborn이 설치되지 않았습니다.")
    print("          Classification 평가를 위해 설치하세요:")
    print("          pip install scikit-learn matplotlib seaborn")


class ClassificationEvaluator:
    """분류 모델 평가 담당 클래스"""
    
    def __init__(self, config: Dict[str, Any], model):
        """
        Args:
            config: 설정 딕셔너리
            model: YOLO 분류 모델
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "Classification 평가를 위해 필요한 패키지가 설치되지 않았습니다.\n"
                "다음 명령으로 설치하세요: pip install scikit-learn matplotlib seaborn"
            )
        
        self.config = config
        self.model = model
        self.product = config['product']
        
    def evaluate(self, val_dir: Optional[str] = None, save_results: bool = True) -> Dict[str, Any]:
        """
        검증 데이터셋에 대해 평가 수행
        
        Args:
            val_dir: 검증 데이터 디렉토리 (None이면 자동 설정)
            save_results: 결과 저장 여부
            
        Returns:
            평가 메트릭 딕셔너리
        """
        # 검증 디렉토리 설정
        if val_dir is None:
            val_dir = self._get_validation_dir()
        
        val_dir = Path(val_dir)
        if not val_dir.exists():
            raise FileNotFoundError(f"검증 디렉토리를 찾을 수 없습니다: {val_dir}")
        
        print(f"[INFO] ==========================================")
        print(f"[INFO] 평가 시작")
        print(f"[INFO] - 검증 데이터: {val_dir}")
        print(f"[INFO] ==========================================")
        
        # 1. 검증 이미지와 라벨 수집
        image_paths, y_true, class_names = self._collect_validation_data(val_dir)
        print(f"[INFO] 총 {len(image_paths)}개 이미지, {len(class_names)}개 클래스")
        
        # 2. 예측
        print(f"[INFO] 예측 중...")
        results = self.model.predict(image_paths, verbose=False)
        
        # 3. 예측 결과 추출
        y_pred = []
        for result in results:
            if hasattr(result, 'probs'):
                y_pred.append(result.probs.top1)
            else:
                # Detection 결과라면 가장 높은 신뢰도의 클래스 사용
                max_conf_class = result.boxes.cls[result.boxes.conf.argmax()].item()
                y_pred.append(int(max_conf_class))
        
        # 4. Confusion Matrix 생성
        cm = confusion_matrix(y_true, y_pred)
        
        # 5. Classification Report
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        
        # 6. 정확도 계산
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"[INFO] ==========================================")
        print(f"[INFO] 평가 완료!")
        print(f"[INFO] - 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"[INFO] ==========================================")
        
        # 7. 결과 저장
        if save_results:
            self._save_confusion_matrix(cm, class_names)
            self._save_report(report, cm, accuracy)
        
        # 8. 콘솔 출력
        print("\n=== Classification Report ===")
        print(report)
        print("\n=== Confusion Matrix ===")
        print(cm)
        
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names,
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
        }
    
    def _collect_validation_data(self, val_dir: Path) -> Tuple[List[str], List[int], List[str]]:
        """
        검증 데이터 수집
        
        Args:
            val_dir: 검증 데이터 디렉토리
            
        Returns:
            (image_paths, y_true, class_names) 튜플
        """
        image_paths = []
        y_true = []
        class_names = sorted(os.listdir(val_dir))
        
        # 디렉토리가 아닌 것 제외
        class_names = [c for c in class_names if (val_dir / c).is_dir()]
        
        img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.ppm')
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = val_dir / class_name
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in img_exts:
                    image_paths.append(str(img_file))
                    y_true.append(class_idx)
        
        return image_paths, y_true, class_names
    
    def _save_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """
        Confusion Matrix 시각화 및 저장
        
        Args:
            cm: Confusion matrix
            class_names: 클래스 이름 리스트
        """
        plt.figure(figsize=(12, 10))
        
        # 히트맵 그리기
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'Confusion Matrix - {self.product.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 저장
        save_path = self._get_output_path('confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Confusion Matrix 저장: {save_path}")
    
    def _save_report(self, report: str, cm: np.ndarray, accuracy: float):
        """
        분류 리포트 텍스트 저장
        
        Args:
            report: Classification report 문자열
            cm: Confusion matrix
            accuracy: 정확도
        """
        save_path = self._get_output_path('classification_report.txt')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Classification Evaluation Report ===\n")
            f.write(f"Product: {self.product}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write("\n")
            f.write("=== Classification Report ===\n\n")
            f.write(report)
            f.write("\n\n")
            f.write("=== Confusion Matrix ===\n")
            f.write(str(cm))
        
        print(f"[INFO] Report 저장: {save_path}")
    
    def _get_validation_dir(self) -> str:
        """
        검증 데이터셋 경로 자동 설정
        
        Returns:
            검증 디렉토리 경로
        """
        paths = self.config['paths']
        product = self.config['product']
        data_version = self.config.get('data_version', 'v1')
        
        return f"{paths['data_root']}/{product}/{data_version}/val"
    
    def _get_output_path(self, filename: str) -> str:
        """
        출력 파일 경로 생성
        
        Args:
            filename: 파일 이름
            
        Returns:
            전체 파일 경로
        """
        paths = self.config['paths']
        product = self.config['product']
        
        # 날짜별 폴더로 관리
        date_str = datetime.now().strftime('%y%m%d')
        output_dir = Path(paths['output_root']) / product / "runs" / "evaluation" / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir / filename)

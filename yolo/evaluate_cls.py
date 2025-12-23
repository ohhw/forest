#!/usr/bin/env python3
"""
Classification 모델 평가 스크립트
Confusion Matrix 및 Classification Report 생성
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from core.config import ConfigLoader
from core.evaluator import ClassificationEvaluator
from ultralytics import YOLO


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Classification 모델 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 평가 (검증 데이터셋)
  python evaluate.py --config configs/models/cls/csn.yaml --weights classify/csn_cls_11s_25121910h/weights/best.pt
  
  # 특정 검증 디렉토리
  python evaluate.py --config configs/models/cls/csn.yaml --weights best.pt --val-dir /path/to/validation
  
  # 결과 저장 안 함 (콘솔만)
  python evaluate.py --config configs/models/cls/csn.yaml --weights best.pt --no-save
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='설정 파일 경로 (Classification 전용)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='학습된 모델 가중치 경로 (.pt 파일)'
    )
    
    parser.add_argument(
        '--val-dir',
        type=str,
        default=None,
        help='검증 데이터 디렉토리 (기본값: 설정 파일에서 자동 설정)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='결과 저장 안 함 (콘솔만 출력)'
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("Classification 모델 평가 시스템")
    print("=" * 60)
    
    try:
        # 1. 설정 로드
        print(f"\n[1/4] 설정 로드 중...")
        config_loader = ConfigLoader(args.config, task='classify')
        config = config_loader.load()
        
        if config['task'] != 'classify':
            print(f"\n❌ Classification 모델만 평가 가능합니다 (현재: {config['task']})")
            sys.exit(1)
        
        print(f"  ✓ Product: {config['product']}")
        print(f"  ✓ Model: {config['model']}")
        
        # 2. 모델 로드
        print(f"\n[2/4] 모델 로드 중...")
        weights_path = Path(args.weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {args.weights}")
        
        model = YOLO(str(weights_path))
        print(f"  ✓ 모델 로드 완료")
        
        # 3. Evaluator 초기화
        print(f"\n[3/4] Evaluator 초기화 중...")
        evaluator = ClassificationEvaluator(config, model)
        
        # 4. 평가 실행
        print(f"\n[4/4] 평가 실행...")
        print("-" * 60)
        
        metrics = evaluator.evaluate(
            val_dir=args.val_dir,
            save_results=not args.no_save
        )
        
        print("-" * 60)
        
        # 5. 완료
        print("\n" + "=" * 60)
        print("✅ 평가 완료!")
        print("=" * 60)
        print(f"📊 Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"📈 클래스 수: {len(metrics['class_names'])}")
        print(f"🖼️  이미지 수: {len(metrics['y_true'])}")
        
        if not args.no_save:
            print(f"\n💾 결과 저장됨:")
            print(f"  - Confusion Matrix (PNG)")
            print(f"  - Classification Report (TXT)")
        
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

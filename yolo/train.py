#!/usr/bin/env python3
"""
YOLO 통합 학습 스크립트
Detection 및 Classification 모두 지원
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from core.config import ConfigLoader
from core.trainer import YOLOTrainer


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='YOLO 모델 학습',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Detection 학습
  python train.py --config configs/models/dod/csn.yaml
  
  # Classification 학습
  python train.py --config configs/models/cls/csn.yaml
  
  # 이전 학습 재개
  python train.py --config configs/models/dod/csn.yaml --resume
  
  # Task 명시적 지정
  python train.py --config configs/models/dod/csn.yaml --task detect
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='설정 파일 경로 (예: configs/models/dod/csn.yaml)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['detect', 'classify'],
        default=None,
        help='Task 타입 (설정 파일에서 자동 감지되지만 명시 가능)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='이전 학습 재개'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='학습 후 검증 실행'
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("YOLO 통합 학습 시스템")
    print("=" * 60)
    
    try:
        # 1. 설정 로드
        print(f"\n[1/4] 설정 로드 중...")
        config_loader = ConfigLoader(args.config, task=args.task)
        config = config_loader.load()
        
        print(f"  ✓ Product: {config['product']}")
        print(f"  ✓ Task: {config['task']}")
        print(f"  ✓ Model: {config['model']}")
        
        # 2. Trainer 초기화
        print(f"\n[2/4] Trainer 초기화 중...")
        trainer = YOLOTrainer(config)
        
        # 3. 모델 로드
        print(f"\n[3/4] 모델 로드 중...")
        model = trainer.setup_model()
        print(f"  ✓ 모델 로드 완료")
        
        # 4. 학습 실행
        print(f"\n[4/7] 학습 실행...")
        print("-" * 60)
        best_weights = trainer.train(model, resume=args.resume)
        print("-" * 60)
        
        # 5. 학습 완료 후 자동 검증 및 지표 출력
        print(f"\n[5/7] 최종 성능 평가 중...")
        print("-" * 60)
        
        # Best 모델로 최종 검증 실행
        best_model = trainer.load_trained_model(best_weights)
        metrics = trainer.validate(best_model, split='val')
        
        # 주요 지표 출력
        print("\n" + "=" * 60)
        print("📊 최종 학습 결과")
        print("=" * 60)
        
        if config['task'] == 'detect':
            # Detection 메트릭
            print("\n🎯 Detection 성능 지표:")
            print(f"  ├─ mAP50-95:  {metrics.box.map:.4f}")
            print(f"  ├─ mAP50:     {metrics.box.map50:.4f}")
            print(f"  ├─ mAP75:     {metrics.box.map75:.4f}")
            print(f"  ├─ Precision: {metrics.box.mp:.4f}")
            print(f"  └─ Recall:    {metrics.box.mr:.4f}")
            
            # 클래스별 mAP (있는 경우)
            if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
                print(f"\n  📋 클래스별 mAP50-95:")
                for i, map_val in enumerate(metrics.box.maps):
                    print(f"     Class {i}: {map_val:.4f}")
        else:
            # Classification 메트릭
            print("\n🏷️  Classification 성능 지표:")
            print(f"  ├─ Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
            print(f"  └─ Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")
        
        print("-" * 60)
        
        # 6. 검증 데이터로 예측 실행 (기존 코드 호환)
        if config['task'] == 'detect':
            # Detection: 신뢰도 포함/제외 2번 실행
            print(f"\n[6/7] 검증 데이터 예측 (신뢰도 포함)...")
            print("-" * 60)
            trainer.predict_on_validation(best_model, show_conf=True)
            print("-" * 60)
            
            print(f"\n[7/7] 검증 데이터 예측 (신뢰도 제외)...")
            print("-" * 60)
            trainer.predict_on_validation(best_model, show_conf=False)
            print("-" * 60)
        else:
            # Classification: 1번만 실행
            print(f"\n[6/6] 검증 데이터 예측...")
            print("-" * 60)
            trainer.predict_on_validation(best_model, show_conf=False)
            print("-" * 60)
        
        # 6. 완료
        print("\n" + "=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)
        print(f"📦 Best weights: {best_weights}")
        print(f"📊 결과 디렉토리: {trainer._get_output_dir()}")
        print(f"\n💾 저장된 파일:")
        print(f"  - weights/best.pt")
        print(f"  - weights/last.pt")
        print(f"  - results.png (학습 곡선)")
        print(f"  - confusion_matrix.png")
        print("\n💡 다음 단계:")
        print(f"  - 추론: python predict.py --config {args.config} --weights {best_weights}")
        if config['task'] == 'classify':
            print(f"  - 평가: python evaluate.py --config {args.config} --weights {best_weights}")
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

#!/usr/bin/env python3
"""
YOLO 통합 추론 스크립트
학습된 모델로 이미지 예측 수행
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from core.config import ConfigLoader
from core.predictor import YOLOPredictor


def find_latest_weights(config: dict) -> str:
    """
    가장 최근 학습된 모델의 best.pt 자동 탐색
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        best.pt 경로
        
    Raises:
        FileNotFoundError: best.pt를 찾을 수 없는 경우
    """
    paths = config['paths']
    product = config['product']
    
    runs_dir = Path(paths['output_root']) / product / "runs"
    
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"runs 디렉토리를 찾을 수 없습니다: {runs_dir}\n"
            f"먼저 학습을 실행하세요: python train.py --config {config.get('_config_path', 'config.yaml')}"
        )
    
    # runs/ 아래의 모든 디렉토리에서 weights/best.pt 찾기
    best_pt_paths = []
    for model_dir in runs_dir.iterdir():
        if model_dir.is_dir():
            best_pt = model_dir / "weights" / "best.pt"
            if best_pt.exists():
                best_pt_paths.append(best_pt)
    
    if not best_pt_paths:
        raise FileNotFoundError(
            f"best.pt를 찾을 수 없습니다: {runs_dir}\n"
            f"먼저 학습을 실행하세요: python train.py --config {config.get('_config_path', 'config.yaml')}"
        )
    
    # 가장 최근 파일 선택 (수정 시간 기준)
    latest = max(best_pt_paths, key=lambda p: p.stat().st_mtime)
    
    return str(latest)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='YOLO 모델 추론',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 추론 (검증 데이터셋)
  python predict.py --config configs/models/dod/csn.yaml --weights runs/detect/csn_dod_11n_25121910h/weights/best.pt
  
  # 특정 이미지 디렉토리
  python predict.py --config configs/models/dod/csn.yaml --weights best.pt --source /path/to/images
  
  # Confidence on/off 두 버전
  python predict.py --config configs/models/dod/csn.yaml --weights best.pt --both-conf
  
  # Confidence threshold 변경
  python predict.py --config configs/models/dod/csn.yaml --weights best.pt --conf 0.7
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='설정 파일 경로'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=False,  # 선택적으로 변경
        default=None,
        help='학습된 모델 가중치 경로 (.pt 파일). 생략 시 자동으로 최신 best.pt 사용'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='추론할 이미지 경로 (파일 또는 디렉토리). 기본값: 검증 데이터셋'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=None,
        help='Confidence threshold (기본값: 설정 파일 사용)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=None,
        help='IoU threshold (기본값: 설정 파일 사용)'
    )
    
    parser.add_argument(
        '--both-conf',
        action='store_true',
        help='Confidence 표시 on/off 두 버전 모두 실행'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='결과 이미지 저장 안 함'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['detect', 'classify'],
        default=None,
        help='Task 타입 (설정 파일에서 자동 감지)'
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("YOLO 통합 추론 시스템")
    print("=" * 60)
    
    try:
        # 1. 설정 로드
        print(f"\n[1/4] 설정 로드 중...")
        config_loader = ConfigLoader(args.config, task=args.task)
        config = config_loader.load()
        
        task = config['task']
        task_name = "🎯 Detection (객체 탐지)" if task == 'detect' else "🏷️  Classification (분류)"
        
        print(f"  ✓ Product: {config['product'].upper()}")
        print(f"  ✓ Task: {task_name}")
        print(f"  ✓ Model: {config['model']}")
        
        # 2. Weights 경로 결정
        if args.weights is None:
            print(f"\n[2/5] 최신 모델 자동 탐색 중...")
            weights_path = find_latest_weights(config)
            print(f"  ✓ 자동 탐색: {weights_path}")
        else:
            weights_path = args.weights
            print(f"\n[2/5] 가중치: {weights_path}")
        
        # 3. Task별 안내
        print(f"\n[3/5] Task 특성")
        if task == 'detect':
            print(f"  📦 Detection 모드:")
          4. Predictor 초기화
        print(f"\n[4/5] Predictor 초기화 중...")
        predictor = YOLOPredictor(config, weights_path)
        print(f"  ✓ 모델 로드 완료")
        
        # 5. 추론 소스 결정
        if args.source is None:
            source = predictor.get_validation_path()
            print(f"  ✓ 추론 소스: 검증 데이터셋")
        else:
            source = args.source
            print(f"  ✓ 추론 소스: {source}")
        
        # 6. 추론 실행
        print(f"\n[5/5
        if args.source is None:
            source = predictor.get_validation_path()
            print(f"  ✓ 추론 소스: 검증 데이터셋")
        else:
            source = args.source
            print(f"  ✓ 추론 소스: {source}")
        
        # 5. 추론 실행
        print(f"\n[4/4] 추론 실행...")
        print("-" * 60)
        
        if args.both_conf:
            # Confidence on/off 두 버전
            results_with, results_without = predictor.predict_with_without_conf(
                source=source,
                conf=args.conf,
            )
            print(f"  ✓ Confidence ON: {len(results_with)}개 이미지")
            print(f"  ✓ Confidence OFF: {len(results_without)}개 이미지")
        else:
            # 일반 추론
            results = predictor.predict(
                source=source,
                save=not args.no_save,
                conf=args.conf,
                iou=args.iou,
            )
            print(f"  ✓ 추론 완료: {len(results)}개 이미지")
        
        print("-" * 60)
        
        # 6. 완료 및 결과 요약
        print("\n" + "=" * 60)
        print("✅ 추론 완료!")
        print("=" * 60)
        
        # Task별 결과 안내
        if task == 'detect':
            print(f"\n📦 Detection 결과:")
            print(f"  - 탐지된 객체 위치 및 클래스")
            print(f"  - Bounding Box가 그려진 이미지")
            print(f"  - Crop된 객체 이미지 (save_crop=True인 경우)")
            print(f"  - 라벨 텍스트 파일 (YOLO 형식)")
        else:
            print(f"\n🏷️  Classification 결과:")
            print(f"  - 예측된 클래스 (등급/색택)")
            print(f"  - 클래스별 confidence score")
            print(f"  - 라벨이 표시된 이미지")
        
        print(f"\n📊 결과 저장 위치:")
        print(f"   {predictor._get_output_dir()}")
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

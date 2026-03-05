#!/usr/bin/env python3
"""
YOLO 하이퍼파라미터 튜닝 스크립트
Ray Tune을 활용한 자동 하이퍼파라미터 최적화
"""

import argparse
import sys
from pathlib import Path
import json
import yaml
from datetime import datetime
import pickle
import shutil

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from core.config import ConfigLoader
from ultralytics import YOLO


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='YOLO 하이퍼파라미터 튜닝',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Detection 튜닝
  python tune.py --config configs/tune/dod_tune.yaml --product csn --iterations 50
  
  # Classification 튜닝
  python tune.py --config configs/tune/cls_tune.yaml --product jjb --iterations 30
  
  # GPU 지정
  python tune.py --config configs/tune/dod_tune.yaml --product csn --device 0
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='튜닝 설정 파일 경로 (configs/tune/*.yaml)'
    )
    
    parser.add_argument(
        '--product',
        type=str,
        required=False,  # 선택적으로 변경
        help='임산물 이름 (설정 파일에 명시되어 있으면 생략 가능)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='튜닝 반복 횟수 (기본값: 설정 파일 사용)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='사용할 GPU 디바이스 (기본값: 0)'
    )
    
    parser.add_argument(
        '--auto-update',
        action='store_true',
        help='튜닝 완료 후 자동으로 설정 파일에 반영'
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("=" * 60)
    print("YOLO 하이퍼파라미터 튜닝 시스템")
    print("=" * 60)
    print("\n⚠️  주의: 튜닝은 시간이 오래 걸립니다!")
    print("         Ray Tune이 설치되어 있어야 합니다.")
    print("         pip install 'ray[tune]'\n")
    
    try:
        # 1. 튜닝 설정 로드
        print(f"\n[1/4] 튜닝 설정 로드 중...")
        tune_config = ConfigLoader.load_yaml(args.config)
        
        # 2. Product 결정 (설정 파일 우선, 없으면 인자 사용)
        if 'product' in tune_config:
            product = tune_config['product']
            print(f"  ✓ Product: {product} (설정 파일에서 자동 감지)")
        elif args.product:
            product = args.product
            print(f"  ✓ Product: {product} (인자로 전달)")
        else:
            raise ValueError("product를 설정 파일에 명시하거나 --product 인자로 전달하세요")
        
        # 3. 임산물 기본 설정 로드
        print(f"\n[2/4] 임산물 설정 로드 중...")
        
        # base_config (단일) 또는 base_configs (딕셔너리) 지원
        if 'base_config' in tune_config:
            base_config_path = tune_config['base_config']
        elif 'base_configs' in tune_config:
            base_config_path = tune_config['base_configs'][product]
        else:
            raise ValueError("튜닝 설정에 base_config 또는 base_configs가 없습니다")
        
        config_loader = ConfigLoader(base_config_path)
        config = config_loader.load()
        
        print(f"  ✓ Product: {config['product']}")
        print(f"  ✓ Task: {config['task']}")
        print(f"  ✓ Model: {config['model']}")
        
        # 3. 모델 로드
        print(f"\n[3/4] 모델 로드 중...")
        model_name = config['model']
        model_path = f"{model_name}.pt"
        model = YOLO(model_path)
        print(f"  ✓ 모델 로드 완료")
        
        # 4. 튜닝 실행
        print(f"\n[4/4] 튜닝 실행...")
        print("-" * 60)
        
        # 데이터 경로
        paths = config['paths']
        product = config['product']
        data_version = config.get('data_version', 'v2')
        
        if config['task'] == 'detect':
            data_path = f"{paths['data_root']}/{product}/{product}_defect_detection_data_{data_version}.yaml"
        else:
            data_path = f"{paths['data_root']}/{product}/{data_version}"
        
        # 튜닝 설정
        tune_settings = tune_config['tune_settings']
        iterations = args.iterations if args.iterations is not None else tune_settings.get('iterations', 30)
        
        print(f"  ✓ Iterations: {iterations}")
        print(f"  ✓ Data: {data_path}")
        print(f"  ✓ GPU: {args.device}")
        
        # 중간 저장 디렉토리 설정
        checkpoint_dir = ROOT / "tune_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{product}_tune_checkpoint.pkl"
        
        print(f"  ✓ Checkpoint: {checkpoint_file}")
        print(f"\n🚀 튜닝 시작 (시간이 오래 걸립니다)...\n")
        
        # 중간 저장을 위한 설정 저장
        checkpoint_data = {
            'config': config,
            'tune_config': tune_config,
            'base_config_path': base_config_path,
            'timestamp_start': datetime.now().isoformat(),
            'iterations': iterations,
            'data_path': data_path
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"  ✓ 중간 저장 설정 완료\n")
        
        # 튜닝 epochs 설정 (tune_settings 우선, 없으면 기본 학습 epochs 사용)
        tune_epochs = tune_settings.get('epochs', config['training']['epochs'])
        
        # YOLO의 tune() 메서드 사용
        results = model.tune(
            data=data_path,
            epochs=tune_epochs,
            iterations=iterations,
            optimizer=config['training'].get('optimizer', 'AdamW'),
            device=args.device,
            use_ray=tune_settings.get('use_ray', True),
        )
        
        print("-" * 60)
        
        # 튜닝 결과를 checkpoint에 저장 (pickle 가능한 형태로 변환)
        try:
            # results를 dict로 변환 (Ray 객체 제거)
            if results and isinstance(results, dict):
                results_serializable = {k: v for k, v in results.items() 
                                       if not k.startswith('_')}
            else:
                results_serializable = str(results) if results else None
            
            checkpoint_data['results'] = results_serializable
            checkpoint_data['timestamp_end'] = datetime.now().isoformat()
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"  ✓ 튜닝 결과 중간 저장 완료\n")
        except Exception as e:
            print(f"  ⚠️  중간 저장 실패 (계속 진행): {e}\n")
        
        # 5. 결과 저장
        print(f"\n[5/6] 튜닝 결과 저장 중...")
        log_path = save_tuning_results(
            config=config,
            tune_config=tune_config,
            results=results,
            base_config_path=base_config_path
        )
        print(f"  ✓ 로그 저장: {log_path}")
        
        # 6. 자동 업데이트 (옵션)
        if args.auto_update:
            print(f"\n[6/6] 설정 파일 자동 업데이트 중...")
            update_config_file(base_config_path, results)
            print(f"  ✓ 설정 파일 업데이트 완료: {base_config_path}")
        
        # 완료
        print("\n" + "=" * 60)
        print("✅ 튜닝 완료!")
        print("=" * 60)
        print(f"📊 최적 하이퍼파라미터:")
        print(results)
        print(f"\n📝 로그 저장 위치: {log_path}")
        print(f"💾 Checkpoint 저장: {checkpoint_file}")
        
        if not args.auto_update:
            print("\n💡 설정 파일에 자동 반영하려면 --auto-update 옵션을 사용하세요")
            print(f"  또는 수동으로 반영하세요: {base_config_path}")
        
        # Checkpoint 백업 (성공 시)
        try:
            if checkpoint_file.exists():
                backup_checkpoint = checkpoint_dir / f"{product}_tune_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                shutil.copy2(checkpoint_file, backup_checkpoint)
                print(f"\n✓ Checkpoint 백업: {backup_checkpoint}")
        except Exception as e:
            print(f"\n⚠️  Checkpoint 백업 실패: {e}")
        
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except ImportError as e:
        if 'ray' in str(e).lower():
            print(f"\n❌ Ray Tune이 설치되지 않았습니다:")
            print(f"   pip install 'ray[tune]'")
        else:
            print(f"\n❌ Import 오류: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시 checkpoint 정보 출력
        try:
            checkpoint_dir = ROOT / "tune_checkpoints"
            print(f"\n💾 중간 저장 파일이 있다면:")
            print(f"   {checkpoint_dir}")
            print(f"\n   복구하려면:")
            print(f"   python tune.py --restore <checkpoint_file>")
        except:
            pass
        
        import traceback
        traceback.print_exc()
        sys.exit(1)


def save_tuning_results(config: dict, tune_config: dict, results: dict, base_config_path: str) -> str:
    """
    튜닝 결과를 로그 디렉토리에 저장
    
    Args:
        config: 임산물 설정
        tune_config: 튜닝 설정
        results: 튜닝 결과
        base_config_path: 기본 설정 파일 경로
        
    Returns:
        저장된 로그 파일 경로
    """
    # 로그 디렉토리 생성
    product = config['product']
    task = config['task']
    task_name = 'dod' if task == 'detect' else 'cls'
    
    log_dir = Path(config['paths']['output_root']) / product / "tune_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 타임스탬프
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # results를 JSON 직렬화 가능한 형태로 변환
    best_params = {}
    if results:
        try:
            if isinstance(results, dict):
                # 이미 dict면 그대로 사용
                best_params = {k: v for k, v in results.items() 
                              if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))}
            elif hasattr(results, '__dict__'):
                # 객체면 속성 추출
                best_params = {k: v for k, v in results.__dict__.items() 
                              if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))}
            else:
                # 변환 불가능하면 문자열로
                best_params = {'result': str(results)}
        except Exception as e:
            best_params = {'error': f'Failed to serialize: {str(e)}', 'raw': str(results)[:500]}
    
    # 로그 데이터 구성
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'product': product,
        'task': task,
        'model': config['model'],
        'data_version': config.get('data_version', 'v2'),
        'config_file': base_config_path,
        'tune_settings': tune_config['tune_settings'],
        'search_space': tune_config['search_space'],
        'best_hyperparameters': best_params,
        'original_training_params': config['training']
    }
    
    # JSON 파일로 저장
    log_file = log_dir / f"{product}_{task_name}_tune_{timestamp}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    return str(log_file)


def update_config_file(config_path: str, results: dict):
    """
    튜닝 결과를 설정 파일에 자동 반영
    
    Args:
        config_path: 업데이트할 설정 파일 경로
        results: 튜닝 결과 딕셔너리
    """
    if not results:
        print("  ⚠️  튜닝 결과가 비어있어 업데이트를 건너뜁니다")
        return
    
    config_path = Path(config_path)
    
    # 기존 설정 파일 읽기
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # training 섹션 업데이트
    if 'training' not in config_data:
        config_data['training'] = {}
    
    # 결과에서 추출 가능한 파라미터만 업데이트
    tunable_params = ['dropout', 'iou', 'lr0', 'lrf', 'batch', 'patience', 'epochs']
    
    updated_params = []
    for param in tunable_params:
        if param in results:
            old_value = config_data['training'].get(param, 'N/A')
            new_value = results[param]
            config_data['training'][param] = new_value
            updated_params.append(f"    - {param}: {old_value} → {new_value}")
    
    # 백업 생성
    backup_path = config_path.with_suffix('.yaml.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    # 원본 파일 업데이트
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    if updated_params:
        print(f"  ✓ 업데이트된 파라미터:")
        print('\n'.join(updated_params))
        print(f"  ✓ 백업 파일: {backup_path}")
    else:
        print("  ℹ️  업데이트할 파라미터가 없습니다")


if __name__ == '__main__':
    main()

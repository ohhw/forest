"""
YOLO 객체 탐지 모델 학습 통합 스크립트
임산물 결함 탐지를 위한 학습
"""

import os
import argparse
import torch.hub
from ultralytics import YOLO
from pathlib import Path


# 제품별 기본 설정
PRODUCT_CONFIGS = {
    'csn': {
        'name': '밤',
        'data_yaml': 'csn_defect_detection_data.yaml',
        'epochs': 150,
        'batch': 32,
        'patience': 50,
        'dropout': 0.25,
        'iou': 0.35,
        'optimizer': 'auto',
        'lr0': None,
        'lrf': None
    },
    'jjb': {
        'name': '건대추',
        'data_yaml': 'jjb_defect_detection_data.yaml',
        'epochs': 300,
        'batch': 32,
        'patience': 100,
        'dropout': 0.3,
        'iou': 0.3,
        'optimizer': 'AdamW',
        'lr0': 0.0005,
        'lrf': 0.00001
    },
    'wln': {
        'name': '호두',
        'data_yaml': 'wln_defect_detection_data.yaml',
        'epochs': 250,
        'batch': 32,
        'patience': 100,
        'dropout': 0.255,
        'iou': 0.415,
        'optimizer': 'auto',
        'lr0': None,
        'lrf': None
    },
    'obj': {
        'name': '일반객체',
        'data_yaml': 'obj_detection_data.yaml',
        'epochs': 200,
        'batch': 32,
        'patience': 75,
        'dropout': 0.3,
        'iou': 0.35,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.00001
    }
}


def train_model(
    product: str,
    model_name: str,
    data_version: str,
    yolo_model: str = '11s',
    user: str = 'hwoh',
    base_path: str = '/hdd/datasets',
    **kwargs
):
    """
    탐지 모델 학습
    
    Args:
        product: 제품 코드 (csn, jjb, wln, obj)
        model_name: 모델 이름 (자동 생성 또는 수동 지정)
        data_version: 데이터 버전 (예: v10)
        yolo_model: YOLO 모델 크기 (11n, 11s, 11m, 11l, 11x)
        user: 사용자 이름
        base_path: 데이터셋 기본 경로
        **kwargs: 추가 학습 파라미터
    """
    
    # 제품 설정 가져오기
    if product not in PRODUCT_CONFIGS:
        raise ValueError(f"지원하지 않는 제품: {product}. 사용 가능: {list(PRODUCT_CONFIGS.keys())}")
    
    config = PRODUCT_CONFIGS[product].copy()
    product_name = config.pop('name')
    data_yaml = config.pop('data_yaml')
    
    # kwargs로 기본 설정 오버라이드
    config.update({k: v for k, v in kwargs.items() if v is not None})
    
    # 경로 설정
    work_dir = f"/home/{user}/detection/{product}"
    data_type = "obj_data" if product == "obj" else "dod_data"
    data_yaml_path = f"{base_path}/{data_type}/{product}/{data_yaml}"
    
    # 데이터 버전 지정된 경우 YAML 경로 수정
    if data_version:
        # v10 버전 형식인 경우
        if data_version.startswith('v'):
            data_yaml_path = f"{base_path}/{data_type}/{product}/{product}_defect_detection_data_{data_version}.yaml"
        else:
            # 날짜 형식인 경우 (예: 251015_psm)
            data_yaml_path = f"{base_path}/{data_type}/{product}/{data_version}/{data_yaml}"
    
    print(f"\n{'='*70}")
    print(f"🎯 {product_name}({product.upper()}) 탐지 모델 학습")
    print(f"{'='*70}")
    print(f"YOLO 모델: yolo{yolo_model}")
    print(f"모델명: {model_name}")
    print(f"데이터 버전: {data_version}")
    print(f"데이터 YAML: {data_yaml_path}")
    print(f"작업 디렉토리: {work_dir}")
    print(f"\n학습 설정:")
    for key, value in config.items():
        if value is not None:
            print(f"  - {key}: {value}")
    print(f"{'='*70}\n")
    
    # 작업 디렉토리 설정
    os.chdir(work_dir)
    torch.hub.set_dir(work_dir)
    
    # 가중치 경로 설정
    weight_path = f"{work_dir}/yolo{yolo_model}.pt"
    if not os.path.exists(weight_path):
        print(f"⚠️  가중치 파일이 없습니다: {weight_path}")
        print(f"💡 자동으로 다운로드됩니다...")
    
    # 모델 로드
    model = YOLO(weight_path)
    print(f"✅ 모델 로드 완료: yolo{yolo_model}.pt\n")
    
    # 학습 파라미터 구성
    train_kwargs = {
        'data': data_yaml_path,
        'name': model_name,
        'exist_ok': True,
        'project': f"{work_dir}/runs/detect"
    }
    
    # None이 아닌 설정만 추가
    train_kwargs.update({k: v for k, v in config.items() if v is not None})
    
    # 학습 시작
    print("🚀 학습 시작...")
    results = model.train(**train_kwargs)
    
    print(f"\n✅ 학습 완료!")
    print(f"📁 결과 저장 위치: {work_dir}/runs/detect/{model_name}")
    
    best_weight_path = f"{work_dir}/runs/detect/{model_name}/weights/best.pt"
    return best_weight_path, model_name


def validate_model(
    model_path: str,
    device: int = 0
):
    """
    모델 검증
    
    Args:
        model_path: 모델 경로
        device: GPU 디바이스 번호
    """
    
    print(f"\n{'='*70}")
    print(f"📊 모델 검증")
    print(f"{'='*70}")
    print(f"모델: {model_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        return
    
    model = YOLO(model_path)
    metrics = model.val(device=device)
    
    print(f"\n📊 검증 결과:")
    print(metrics)
    print(f"\n✅ 검증 완료!")


def predict_model(
    model_path: str,
    images_dir: str,
    output_name: str,
    work_dir: str,
    model_name: str,
    conf: float = 0.5,
    iou: float = 0.3,
    show_conf: bool = True,
    save_crop: bool = True,
    save_txt: bool = True,
    **kwargs
):
    """
    모델 예측
    
    Args:
        model_path: 모델 경로
        images_dir: 예측할 이미지 디렉토리
        output_name: 출력 폴더 이름
        work_dir: 작업 디렉토리
        model_name: 모델 이름
        conf: 신뢰도 임계값
        iou: IoU 임계값
        show_conf: 신뢰도 표시 여부
        save_crop: 잘라낸 객체 저장 여부
        save_txt: 텍스트 결과 저장 여부
        **kwargs: 추가 예측 파라미터
    """
    
    print(f"\n{'='*70}")
    print(f"🔮 모델 예측")
    print(f"{'='*70}")
    print(f"모델: {model_path}")
    print(f"이미지: {images_dir}")
    print(f"신뢰도 임계값: {conf}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ 이미지 디렉토리가 존재하지 않습니다: {images_dir}")
        return
    
    model = YOLO(model_path)
    
    # 예측 파라미터 구성
    predict_kwargs = {
        'save': True,
        'save_crop': save_crop,
        'save_txt': save_txt,
        'conf': conf,
        'iou': iou,
        'show_conf': show_conf,
        'exist_ok': True,
        'project': f"{work_dir}/runs/detect/{model_name}",
        'name': output_name
    }
    
    # 추가 파라미터 병합
    predict_kwargs.update({k: v for k, v in kwargs.items() if v is not None})
    
    # 예측 실행
    print("🔮 예측 수행 중...")
    results = model.predict(images_dir, **predict_kwargs)
    
    print(f"\n✅ 예측 완료!")
    print(f"📁 결과 저장: {work_dir}/runs/detect/{model_name}/{output_name}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 탐지 모델 학습 통합 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 학습
  python train_detection.py --product jjb --model-name jjb_test --data-version v10 --train
  
  # 학습 + 검증 + 예측
  python train_detection.py --product csn --model-name csn_test --data-version v8 \
    --train --validate --predict --images-dir /hdd/datasets/dod_data/csn/v8/val/images
  
  # 예측만 수행
  python train_detection.py --product wln --model-name wln_existing \
    --predict --model-path /path/to/best.pt \
    --images-dir /hdd/datasets/dod_data/wln/val2/images
  
  # 사용자 정의 설정
  python train_detection.py --product jjb --model-name jjb_custom \
    --data-version v10 --yolo-model 11l --epochs 200 --batch 16 --train
        """
    )
    
    # 필수 인자
    parser.add_argument(
        '--product', '-p',
        type=str,
        required=True,
        choices=['csn', 'jjb', 'wln', 'obj'],
        help='제품 코드 (csn:밤, jjb:건대추, wln:호두, obj:일반객체)'
    )
    parser.add_argument(
        '--model-name', '-n',
        type=str,
        required=True,
        help='모델 이름 (예: jjb_dod_11s_25071510h)'
    )
    
    # 선택 인자
    parser.add_argument(
        '--data-version', '-v',
        type=str,
        help='데이터 버전 (예: v10, 251015_psm)'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='11s',
        help='YOLO 모델 크기 (11n, 11s, 11m, 11l, 11x, 기본: 11s)'
    )
    parser.add_argument(
        '--user',
        type=str,
        default='hwoh',
        help='사용자 이름 (기본: hwoh)'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='/hdd/datasets',
        help='데이터셋 기본 경로 (기본: /hdd/datasets)'
    )
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, help='학습 에포크')
    parser.add_argument('--batch', type=int, help='배치 크기')
    parser.add_argument('--patience', type=int, help='조기 종료 patience')
    parser.add_argument('--dropout', type=float, help='드롭아웃 비율')
    parser.add_argument('--iou', type=float, help='IoU 임계값')
    parser.add_argument('--optimizer', type=str, help='옵티마이저')
    parser.add_argument('--lr0', type=float, help='초기 학습률')
    parser.add_argument('--lrf', type=float, help='최종 학습률 비율')
    
    # 실행 모드
    parser.add_argument('--train', action='store_true', help='학습 수행')
    parser.add_argument('--validate', action='store_true', help='검증 수행')
    parser.add_argument('--predict', action='store_true', help='예측 수행')
    
    # 예측 관련
    parser.add_argument('--model-path', type=str, help='사용할 모델 경로 (학습 안 할 경우)')
    parser.add_argument('--images-dir', type=str, help='예측할 이미지 디렉토리')
    parser.add_argument('--conf', type=float, default=0.5, help='신뢰도 임계값 (기본: 0.5)')
    parser.add_argument('--no-show-conf', action='store_true', help='신뢰도 숨기기')
    parser.add_argument('--no-save-crop', action='store_true', help='잘라낸 객체 저장 안 함')
    parser.add_argument('--no-save-txt', action='store_true', help='텍스트 결과 저장 안 함')
    
    args = parser.parse_args()
    
    # 모드 검증
    if not (args.train or args.validate or args.predict):
        parser.error("최소 하나의 실행 모드를 선택해야 합니다: --train, --validate, --predict")
    
    # 경로 설정
    work_dir = f"/home/{args.user}/detection/{args.product}"
    model_path = args.model_path
    
    # 학습 수행
    if args.train:
        train_kwargs = {
            'epochs': args.epochs,
            'batch': args.batch,
            'patience': args.patience,
            'dropout': args.dropout,
            'iou': args.iou,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'lrf': args.lrf
        }
        
        model_path, model_name = train_model(
            product=args.product,
            model_name=args.model_name,
            data_version=args.data_version,
            yolo_model=args.yolo_model,
            user=args.user,
            base_path=args.base_path,
            **train_kwargs
        )
    else:
        model_name = args.model_name
        if not model_path:
            # 기본 경로 추정
            model_path = f"{work_dir}/runs/detect/{model_name}/weights/best.pt"
    
    # 검증 수행
    if args.validate:
        validate_model(model_path)
    
    # 예측 수행
    if args.predict:
        if not args.images_dir:
            parser.error("--predict 모드에서는 --images-dir이 필요합니다")
        
        # 신뢰도 표시 여부에 따라 출력 이름 설정
        if args.no_show_conf:
            output_name = f"pred_{model_name}_without_conf"
        else:
            output_name = f"pred_{model_name}"
        
        predict_model(
            model_path=model_path,
            images_dir=args.images_dir,
            output_name=output_name,
            work_dir=work_dir,
            model_name=model_name,
            conf=args.conf,
            iou=args.iou if args.iou else 0.3,
            show_conf=not args.no_show_conf,
            save_crop=not args.no_save_crop,
            save_txt=not args.no_save_txt
        )


if __name__ == "__main__":
    main()

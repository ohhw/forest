"""
YOLO 이미지 분류 모델 학습 통합 스크립트
임산물 색택 분류를 위한 전이학습
"""

import os
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# 제품별 기본 설정
PRODUCT_CONFIGS = {
    'csn': {
        'name': '밤',
        'data_path': '/hdd/datasets/cls_data/csn',
        'val_path': '/hdd/datasets/cls_data/csn/validation',
        'epochs': 200,
        'batch': 64,
        'optimizer': 'SGD',
        'dropout': 0.3,
        'cos_lr': True,
        'lr0': 0.003,
        'weight_decay': 0.01,
        'warmup_epochs': 10.0,
        'patience': 100
    },
    'jjb': {
        'name': '건대추',
        'data_path': '/hdd/datasets/cls_data/jjb',
        'val_path': '/hdd/datasets/cls_data/jjb/validation',
        'epochs': 100,
        'batch': 128,
        'optimizer': 'auto',
        'dropout': None,
        'cos_lr': None,
        'lr0': None,
        'weight_decay': None,
        'warmup_epochs': None,
        'patience': 50
    },
    'wln': {
        'name': '호두',
        'data_path': '/hdd/datasets/cls_data/wln',
        'val_path': '/hdd/datasets/cls_data/wln/validation',
        'epochs': 150,
        'batch': 64,
        'optimizer': 'AdamW',
        'dropout': 0.25,
        'cos_lr': True,
        'lr0': 0.001,
        'weight_decay': 0.005,
        'warmup_epochs': 5.0,
        'patience': 75
    }
}


def train_model(
    product: str,
    model_name: str,
    yolo_model: str = "yolo11s-cls.pt",
    user: str = "hwoh",
    **kwargs
):
    """
    분류 모델 학습
    
    Args:
        product: 제품 코드 (csn, jjb, wln)
        model_name: 모델 이름 (예: csn_25020717h_check)
        yolo_model: 사전학습 모델 (기본: yolo11s-cls.pt)
        user: 사용자 이름
        **kwargs: 추가 학습 파라미터 (기본 설정 오버라이드)
    """
    
    # 제품 설정 가져오기
    if product not in PRODUCT_CONFIGS:
        raise ValueError(f"지원하지 않는 제품: {product}. 사용 가능: {list(PRODUCT_CONFIGS.keys())}")
    
    config = PRODUCT_CONFIGS[product].copy()
    product_name = config.pop('name')
    data_path = config.pop('data_path')
    val_path = config.pop('val_path')
    
    # kwargs로 기본 설정 오버라이드
    config.update({k: v for k, v in kwargs.items() if v is not None})
    
    print(f"\n{'='*70}")
    print(f"🎯 {product_name}({product.upper()}) 분류 모델 학습")
    print(f"{'='*70}")
    print(f"모델: {yolo_model}")
    print(f"모델명: {model_name}")
    print(f"데이터: {data_path}")
    print(f"학습 설정:")
    for key, value in config.items():
        if value is not None:
            print(f"  - {key}: {value}")
    print(f"{'='*70}\n")
    
    # GPU 캐시 정리
    torch.cuda.empty_cache()
    
    # 모델 로드
    model = YOLO(yolo_model)
    print(f"✅ 사전학습 모델 로드 완료: {yolo_model}\n")
    
    # 학습 시작
    print("🚀 학습 시작...")
    start_time = time.time()
    
    # None이 아닌 설정만 전달
    train_kwargs = {
        'data': data_path,
        'name': model_name,
        'imgsz': 224
    }
    train_kwargs.update({k: v for k, v in config.items() if v is not None})
    
    model.train(**train_kwargs)
    
    end_time = time.time()
    
    # 학습 시간 계산
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"\n✅ 학습 완료!")
    print(f"⏱️  학습 시간: {hours}시간 {minutes}분 {seconds}초")
    
    return model, model_name


def evaluate_model(
    model_path: str,
    val_dir: str,
    model_name: str,
    user: str = "hwoh",
    save_dir: str = None
):
    """
    모델 평가 및 결과 저장
    
    Args:
        model_path: 학습된 모델 경로 (best.pt)
        val_dir: 검증 데이터 디렉토리
        model_name: 모델 이름
        user: 사용자 이름
        save_dir: 결과 저장 디렉토리 (None이면 자동 설정)
    """
    
    print(f"\n{'='*70}")
    print(f"📊 모델 평가 시작")
    print(f"{'='*70}")
    print(f"모델: {model_path}")
    print(f"검증 데이터: {val_dir}")
    print(f"{'='*70}\n")
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 검증 데이터 경로 확인
    if not os.path.exists(val_dir):
        print(f"❌ 검증 데이터 경로가 존재하지 않습니다: {val_dir}")
        return
    
    # 이미지 경로와 실제 라벨 수집
    image_paths = []
    y_true = []
    class_names = sorted(os.listdir(val_dir))
    
    print(f"📋 클래스: {', '.join(class_names)}")
    print(f"🔍 이미지 수집 중...\n")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_images = 0
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                y_true.append(class_idx)
                class_images += 1
        
        print(f"  {class_name}: {class_images}개")
    
    print(f"\n📊 총 검증 이미지: {len(image_paths)}개")
    
    # 예측 수행
    print(f"🔮 예측 수행 중...")
    pred_name = f'pred_{model_name}'
    results = model.predict(image_paths, save=True, save_txt=True, name=pred_name)
    
    # 예측 결과 수집
    y_pred = []
    for result in results:
        if hasattr(result, 'probs'):
            y_pred.append(result.probs.top1)
        else:
            # detection 작업인 경우 폴백
            max_conf_class = result.boxes.cls[result.boxes.conf.argmax()].item()
            y_pred.append(int(max_conf_class))
    
    # Confusion Matrix 생성
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*70}")
    print("📊 Confusion Matrix:")
    print(f"{'='*70}")
    print(cm)
    
    # 저장 디렉토리 설정
    if save_dir is None:
        save_dir = f'/home/{user}/classification/classify/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    
    cm_path = os.path.join(save_dir, 'confusion_matrix_viz.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"💾 Confusion Matrix 저장: {cm_path}")
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\n{'='*70}")
    print("📋 Classification Report:")
    print(f"{'='*70}")
    print(report)
    
    # Report 저장
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Validation Dir: {val_dir}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write(f"\n\nClassification Report:\n")
        f.write(report)
    print(f"💾 Report 저장: {report_path}")
    
    print(f"\n✅ 평가 완료!")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 분류 모델 학습 통합 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 설정으로 학습
  python train_classification.py --product csn --model-name csn_25020717h
  
  # 사용자 지정 설정
  python train_classification.py --product jjb --model-name jjb_test \
    --epochs 50 --batch 32 --yolo-model yolo11n-cls.pt
  
  # 학습 + 평가
  python train_classification.py --product csn --model-name csn_test --evaluate
  
  # 평가만 수행
  python train_classification.py --product csn --model-name csn_existing \
    --evaluate-only --model-path /path/to/best.pt
        """
    )
    
    # 필수 인자
    parser.add_argument(
        '--product', '-p',
        type=str,
        required=True,
        choices=['csn', 'jjb', 'wln'],
        help='제품 코드 (csn:밤, jjb:건대추, wln:호두)'
    )
    parser.add_argument(
        '--model-name', '-n',
        type=str,
        required=True,
        help='모델 이름 (예: csn_25020717h)'
    )
    
    # 선택 인자
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolo11s-cls.pt',
        help='사전학습 YOLO 모델 (기본: yolo11s-cls.pt)'
    )
    parser.add_argument(
        '--user',
        type=str,
        default='hwoh',
        help='사용자 이름 (기본: hwoh)'
    )
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, help='학습 에포크')
    parser.add_argument('--batch', type=int, help='배치 크기')
    parser.add_argument('--optimizer', type=str, help='옵티마이저')
    parser.add_argument('--dropout', type=float, help='드롭아웃 비율')
    parser.add_argument('--lr0', type=float, help='초기 학습률')
    parser.add_argument('--weight-decay', type=float, help='가중치 감소')
    parser.add_argument('--warmup-epochs', type=float, help='워밍업 에포크')
    parser.add_argument('--patience', type=int, help='조기 종료 patience')
    parser.add_argument('--cos-lr', action='store_true', help='Cosine 학습률 스케줄')
    
    # 평가 옵션
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='학습 후 자동으로 평가 수행'
    )
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='학습 없이 평가만 수행'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='평가할 모델 경로 (evaluate-only 모드에서 필수)'
    )
    
    args = parser.parse_args()
    
    # 평가만 수행하는 경우
    if args.evaluate_only:
        if not args.model_path:
            parser.error("--evaluate-only 모드에서는 --model-path가 필요합니다")
        
        config = PRODUCT_CONFIGS[args.product]
        evaluate_model(
            model_path=args.model_path,
            val_dir=config['val_path'],
            model_name=args.model_name,
            user=args.user
        )
        return
    
    # 학습 수행
    train_kwargs = {
        'epochs': args.epochs,
        'batch': args.batch,
        'optimizer': args.optimizer,
        'dropout': args.dropout,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience
    }
    
    if args.cos_lr:
        train_kwargs['cos_lr'] = True
    
    model, model_name = train_model(
        product=args.product,
        model_name=args.model_name,
        yolo_model=args.yolo_model,
        user=args.user,
        **train_kwargs
    )
    
    # 학습 후 평가
    if args.evaluate:
        model_path = f'/home/{args.user}/classification/classify/{model_name}/weights/best.pt'
        config = PRODUCT_CONFIGS[args.product]
        evaluate_model(
            model_path=model_path,
            val_dir=config['val_path'],
            model_name=model_name,
            user=args.user
        )


if __name__ == "__main__":
    main()

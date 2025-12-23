#!/bin/bash
# CSN Classification 학습 및 평가 예제

echo "=========================================="
echo "CSN (밤) Classification 학습 및 평가 예제"
echo "=========================================="

# 1. 학습
python train.py --config configs/models/cls/csn.yaml

# 학습이 완료되면 자동으로 best.pt 경로가 출력됩니다
# 예: /home/hwoh/classification/csn/runs/classify/csn_cls_11s_25121910h/weights/best.pt

# 2. 평가 (Confusion Matrix 생성)
WEIGHTS="/home/hwoh/classification/csn/runs/csn_cls_11s_25121910h/weights/best.pt"

python evaluate.py \
  --config configs/models/cls/csn.yaml \
  --weights "$WEIGHTS"

# 3. 추론 (선택)
python predict.py \
  --config configs/models/cls/csn.yaml \
  --weights "$WEIGHTS"

echo "=========================================="
echo "완료!"
echo "Confusion Matrix 확인:"
echo "  /home/hwoh/classification/csn/runs/evaluation/"
echo "=========================================="

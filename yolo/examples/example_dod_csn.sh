#!/bin/bash
# CSN Detection 학습 예제

echo "=========================================="
echo "CSN (밤) Detection 학습 예제"
echo "=========================================="

# 1. 학습
python train.py --config configs/models/dod/csn.yaml

# 학습이 완료되면 자동으로 best.pt 경로가 출력됩니다
# 예: /home/hwoh/detection/csn/detect/csn_dod_11n_25121910h/weights/best.pt

# 2. 추론 (Confidence on/off 두 버전)
WEIGHTS="/home/hwoh/detection/csn/detect/csn_dod_11n_25121910h/weights/best.pt"

python predict.py \
  --config configs/models/dod/csn.yaml \
  --weights "$WEIGHTS" \
  --both-conf

echo "=========================================="
echo "완료!"
echo "=========================================="

#!/bin/bash
# 모든 임산물 학습 자동화 스크립트

echo "=========================================="
echo "모든 임산물 학습 자동화"
echo "=========================================="

# Detection 학습
echo ""
echo "[1/5] CSN Detection 학습..."
python train.py --config configs/models/dod/csn.yaml

echo ""
echo "[2/5] JJB Detection 학습..."
python train.py --config configs/models/dod/jjb.yaml

echo ""
echo "[3/5] WLN Detection 학습..."
python train.py --config configs/models/dod/wln.yaml

# Classification 학습
echo ""
echo "[4/5] CSN Classification 학습..."
python train.py --config configs/models/cls/csn.yaml

echo ""
echo "[5/5] JJB Classification 학습..."
python train.py --config configs/models/cls/jjb.yaml

echo ""
echo "=========================================="
echo "모든 학습 완료!"
echo "=========================================="

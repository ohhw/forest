#!/bin/bash
# 하이퍼파라미터 튜닝 예제

echo "=========================================="
echo "CSN Detection 하이퍼파라미터 튜닝 예제"
echo "=========================================="
echo ""
echo "⚠️  주의사항:"
echo "  - Ray Tune이 설치되어 있어야 합니다: pip install 'ray[tune]'"
echo "  - 튜닝은 시간이 오래 걸립니다 (수 시간 ~ 수 일)"
echo "  - GPU가 충분한지 확인하세요"
echo ""

# CSN Detection 튜닝 (50회 반복)
python tune.py \
  --config configs/tune/dod_tune.yaml \
  --product csn \
  --iterations 50 \
  --device 0

echo ""
echo "=========================================="
echo "튜닝 완료!"
echo "최적 하이퍼파라미터를 configs/models/dod/csn.yaml에 반영하세요"
echo "=========================================="

#!/bin/bash
# Detection과 Classification 비교 예제

echo "=========================================="
echo "🎯 Detection vs 🏷️  Classification 비교"
echo "=========================================="

echo ""
echo "📋 설정 정보 확인"
echo "------------------------------------------"

# Detection 설정 확인
echo ""
echo "1️⃣  Detection (객체 탐지) - CSN"
python -c "
from core.config import ConfigLoader
config = ConfigLoader('configs/models/dod/csn.yaml').load()
print(f'   Task: {config[\"task\"]} (detect)')
print(f'   목적: 이미지에서 결함 위치 탐지')
print(f'   출력: Bounding Box + 클래스')
print(f'   Model: {config[\"model\"]}')
print(f'   데이터: {config[\"paths\"][\"data_root\"]}/{config[\"product\"]}')
"

# Classification 설정 확인
echo ""
echo "2️⃣  Classification (분류) - CSN"
python -c "
from core.config import ConfigLoader
config = ConfigLoader('configs/models/cls/csn.yaml').load()
print(f'   Task: {config[\"task\"]} (classify)')
print(f'   목적: 이미지 전체를 등급/색택으로 분류')
print(f'   출력: 클래스 라벨')
print(f'   Model: {config[\"model\"]}')
print(f'   데이터: {config[\"paths\"][\"data_root\"]}/{config[\"product\"]}')
"

echo ""
echo "=========================================="
echo "📊 주요 차이점"
echo "=========================================="
echo ""
echo "🎯 Detection (탐지):"
echo "  ✓ 여러 객체를 동시에 탐지 가능"
echo "  ✓ 각 객체의 위치 (x, y, w, h)"
echo "  ✓ 객체별 클래스와 confidence"
echo "  ✓ Crop 이미지 저장 가능"
echo "  ✓ 용도: 결함 검출, 위치 파악"
echo ""
echo "🏷️  Classification (분류):"
echo "  ✓ 이미지 하나 = 하나의 클래스"
echo "  ✓ 전체 이미지에 대한 판단"
echo "  ✓ Top-1 클래스 예측"
echo "  ✓ 클래스별 확률값 제공"
echo "  ✓ 용도: 등급 분류, 색택 판정"
echo ""
echo "=========================================="

# 실제 사용 예시
echo ""
echo "💡 실제 사용 예시:"
echo ""
echo "Detection 추론:"
echo "  python predict.py --config configs/models/dod/csn.yaml --weights [WEIGHTS]"
echo ""
echo "Classification 추론:"
echo "  python predict.py --config configs/models/cls/csn.yaml --weights [WEIGHTS]"
echo ""
echo "=========================================="

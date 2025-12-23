#!/bin/bash
# 학습 후 자동 지표 출력 테스트

echo "=========================================="
echo "📊 학습 후 자동 지표 출력 데모"
echo "=========================================="

echo ""
echo "✨ 개선 사항:"
echo "  1. 학습 완료 후 자동으로 최종 검증 실행"
echo "  2. 주요 성능 지표 자동 출력"
echo "  3. Detection: mAP50-95, Precision, Recall 등"
echo "  4. Classification: Top-1, Top-5 Accuracy"
echo ""

echo "🎯 Detection 학습 예시 출력:"
echo "------------------------------------------"
cat << 'DEMO'

[5/5] 최종 성능 평가 중...
------------------------------------------------------------

============================================================
📊 최종 학습 결과
============================================================

🎯 Detection 성능 지표:
  ├─ mAP50-95:  0.8542
  ├─ mAP50:     0.9231
  ├─ mAP75:     0.8876
  ├─ Precision: 0.8934
  └─ Recall:    0.8654

  📋 클래스별 mAP50-95:
     Class 0: 0.8821
     Class 1: 0.8263

------------------------------------------------------------

============================================================
✅ 학습 완료!
============================================================
📦 Best weights: /home/hwoh/detection/csn/runs/csn_dod_11n_25121910h/weights/best.pt
📊 결과 디렉토리: /home/hwoh/detection/csn/runs

💾 저장된 파일:
  - weights/best.pt
  - weights/last.pt
  - results.png (학습 곡선)
  - confusion_matrix.png

💡 다음 단계:
  - 추론: python predict.py --config configs/models/dod/csn.yaml --weights [...]
============================================================
DEMO

echo ""
echo "🏷️  Classification 학습 예시 출력:"
echo "------------------------------------------"
cat << 'DEMO'

[5/5] 최종 성능 평가 중...
------------------------------------------------------------

============================================================
📊 최종 학습 결과
============================================================

🏷️  Classification 성능 지표:
  ├─ Top-1 Accuracy: 0.9342 (93.42%)
  └─ Top-5 Accuracy: 0.9887 (98.87%)

------------------------------------------------------------

============================================================
✅ 학습 완료!
============================================================
📦 Best weights: /home/hwoh/classification/csn/runs/csn_cls_11s_25121910h/weights/best.pt
📊 결과 디렉토리: /home/hwoh/classification/csn/runs

💾 저장된 파일:
  - weights/best.pt
  - weights/last.pt
  - results.png (학습 곡선)
  - confusion_matrix.png

💡 다음 단계:
  - 추론: python predict.py --config configs/models/cls/csn.yaml --weights [...]
  - 평가: python evaluate.py --config configs/models/cls/csn.yaml --weights [...]
============================================================
DEMO

echo ""
echo "=========================================="
echo "📌 주요 지표 설명"
echo "=========================================="
echo ""
echo "🎯 Detection:"
echo "  • mAP50-95: IoU 0.5~0.95 평균 정밀도 (주요 지표)"
echo "  • mAP50:    IoU 0.5에서의 평균 정밀도"
echo "  • mAP75:    IoU 0.75에서의 평균 정밀도"
echo "  • Precision: 정밀도 (예측한 것 중 맞은 비율)"
echo "  • Recall:   재현율 (실제 있는 것 중 찾은 비율)"
echo ""
echo "🏷️  Classification:"
echo "  • Top-1 Accuracy: 1순위 예측 정확도"
echo "  • Top-5 Accuracy: 상위 5개 중 정답 포함 비율"
echo ""
echo "=========================================="
echo ""
echo "✅ 이제 학습만 실행하면 자동으로 지표가 출력됩니다!"
echo "   python train.py --config configs/models/dod/csn.yaml"
echo ""
echo "=========================================="

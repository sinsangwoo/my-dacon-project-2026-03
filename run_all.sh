#!/bin/bash

# 로그 파일 생성
LOG_FILE="overnight_process.log"
echo "================================================" >> $LOG_FILE
echo "🚀 대회 완주 밤샘 파이프라인 시작: $(date)" >> $LOG_FILE
echo "================================================" >> $LOG_FILE

# 1. 특징 추출 (Phase 1: DINOv2 + CLIP)
echo "[1/5] DINOv2 및 CLIP 특징 추출 중..." | tee -a $LOG_FILE
py src/extract_features.py --model all >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 특징 추출 완료" | tee -a $LOG_FILE
else
    echo "❌ 특징 추출 실패. 로그를 확인하세요." | tee -a $LOG_FILE
    exit 1
fi

# 2. LightGBM 학습 및 Optuna 최적화 (Phase 1: ML 모델)
echo "[2/5] LightGBM 학습 및 하이퍼파라미터 튜닝 중..." | tee -a $LOG_FILE
py src/train_lgbm.py --feature all --n_trials 50 >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ LightGBM 학습 완료" | tee -a $LOG_FILE
else
    echo "❌ LightGBM 학습 실패. 로그를 확인하세요." | tee -a $LOG_FILE
    exit 1
fi

# 3. 딥러닝 모델 학습 (Phase 2: Dual-Stream EfficientNet-B0)
echo "[3/5] Dual-Stream EfficientNet 학습 중 (이 과정이 가장 오래 걸립니다)..." | tee -a $LOG_FILE
py src/train.py --epochs 20 --batch_size 16 >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 딥러닝 모델 학습 완료" | tee -a $LOG_FILE
else
    echo "❌ 딥러닝 모델 학습 실패. 로그를 확인하세요." | tee -a $LOG_FILE
    exit 1
fi

# 4. 딥러닝 모델 추론 (Best Folds)
echo "[4/5] 학습된 딥러닝 모델로 테스트 데이터 추론 중..." | tee -a $LOG_FILE
py src/predict.py >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 딥러닝 추론 완료" | tee -a $LOG_FILE
else
    echo "❌ 딥러닝 추론 실패. 로그를 확인하세요." | tee -a $LOG_FILE
    exit 1
fi

# 5. 최종 앙상블 (Phase 3: Geometric Mean Ensemble)
echo "[5/5] 최종 결과 앙상블 및 제출 파일 생성 중..." | tee -a $LOG_FILE
py src/ensemble.py --method geometric --output_csv final_submission.csv >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 최종 앙상블 완료! final_submission.csv가 생성되었습니다." | tee -a $LOG_FILE
else
    echo "❌ 앙상블 실패. 로그를 확인하세요." | tee -a $LOG_FILE
    exit 1
fi

echo "================================================" >> $LOG_FILE
echo "🏁 모든 공정 완료: $(date)" >> $LOG_FILE
echo "================================================" >> $LOG_FILE
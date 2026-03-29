# 🏗️ Structural Stability Classification v9

> **Dacon: 구조물 안정성 분류 경진대회**
> 본 프로젝트는 2D 이미지를 활용하여 구조물의 안정성(Stable/Unstable)을 분류하는 딥러닝/머신러닝 앙상블 파이프라인입니다.

## 🌟 Key Features

- **Dual-Stream EfficientNet-B0**: Front 및 Top 뷰 이미지를 동시에 처리하는 샴 네트워크(Siamese-like) 구조.
- **Foundation Model Features**: DINOv2 및 CLIP(ViT-B/32)을 활용한 고차원 시각 특징 추출.
- **Hybrid Ensemble**: 고성능 딥러닝 모델과 기계학습(LightGBM + Optuna)의 기하 평균(Geometric Mean) 앙상블.
- **Robust Training**: SAM(Sharpness-Aware Minimization), Focal Loss, Physics Consistency Loss(좌우 반전 일관성) 적용.
- **Ready-to-Run**: 전처리부터 최종 제출 파일 생성까지 자동화된 `run_all.sh` 파이프라인 제공.

## 🛠️ Tech Stack

- **Core**: Python 3.10+, PyTorch 2.1+, LightGBM
- **Feature Extraction**: DINOv2, OpenAI CLIP
- **Optimization**: Optuna, SAM Optimizer
- **Augmentation**: Albumentations, Torchvision Transforms

## 📂 Project Structure

```text
.
├── src/
│   ├── dataset.py          # Data Loading & Augmentation
│   ├── model.py            # Dual-Stream EfficientNet & Losses
│   ├── extract_features.py # DINOv2/CLIP Feature Extraction
│   ├── train.py            # Main Deep Learning Training
│   ├── train_lgbm.py       # ML Pipeline with Optuna
│   ├── predict.py          # DL Model Inference & TTA
│   └── ensemble.py         # Final Weighting & Prediction
├── data/                   # (Private) Dataset Directory
├── checkpoints/            # Model Weights & Results
├── run_all.sh              # Full Pipeline Automation Script
└── README.md
```

## 🚀 Getting Started

### 1. Environment Setup
```bash
# 가상환경 생성 및 활성화
python -m venv .myenv
source .myenv/Scripts/activate  # Windows: .myenv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Run Full Pipeline
모든 과정을 하나의 스크립트로 실행할 수 있습니다.
```bash
chmod +x run_all.sh
./run_all.sh
```
*스크립트 수행 내용: 특징 추출 → LightGBM 최적화 → 딥러닝 모델 학습 → 추론 → 최종 앙상블*

## 📈 Results & Analysis

- **Local CV (5-Fold)**: LogLoss 0.02 ~ 0.04
- **Public Leaderboard**: 0.205 (Personal Best)
- **Final Run**: 0.262

### Generalization Insights
- **Identity Leakage 방지**: StratifiedKFold 대신 동일 객체 분리를 위한 검증 전략 필요성 확인.
- **Calibration**: Temperature Scaling 및 Label Smoothing을 통한 Overconfidence 제어.
- **Feature Analysis**: CLIP/DINO 특징의 강력한 식별력을 활용한 안정성 추론.

## 📄 License
본 프로젝트는 Dacon 경진대회 참가를 목적으로 제작되었습니다.

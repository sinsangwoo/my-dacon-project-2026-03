# 🚀 Dacon Structural Stability — Multi-Fold Automation Script (v6.4)
# ------------------------------------------------------------------
# Usage: .\run_all_folds.ps1
# 功能: 1-5 폴드 순차 학습 + 리소스 최적화 대기 + 자동 앙상블 + 이어하기(Resume)

$DataDir   = "data"
$SaveDir   = "checkpoints"
$Epochs    = 30
$BatchSize = 16
$ModelV    = "v6.3_0.1_pursuit"
$Cooldown  = 120  # 에폭 간 대기 시간 (초)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Dacon Multi-Fold Automation Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Fold 1~5 순차 학습
for ($fold=1; $fold -le 5; $fold++) {
    $ckptPath = "$SaveDir\best_fold$fold.pth"
    
    if (Test-Path $ckptPath) {
        Write-Host "[CHECK] Fold $fold checkpoint already exists at $ckptPath." -ForegroundColor Green
        Write-Host "        Skipping to next fold (Resume logic active)." -ForegroundColor Green
        continue
    }

    Write-Host "`n[START] Starting Training for Fold $fold..." -ForegroundColor Yellow
    Write-Host "        Running: py src\train.py --epochs $Epochs --batch_size $BatchSize --fold_idx $fold"
    
    # 학습 실행
    py src\train.py --epochs $Epochs --batch_size $BatchSize --fold_idx $fold --model_v $ModelV
    
    # 종료 코드 확인
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[ERROR] Fold $fold failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Write-Host "        Process Terminated. Please check the logs." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host "[DONE] Fold $fold successfully completed." -ForegroundColor Green
    
    if ($fold -lt 5) {
        Write-Host "[WAIT] Cooling down for $Cooldown seconds to stabilize CPU/Memory..." -ForegroundColor Gray
        Start-Sleep -Seconds $Cooldown
    }
}

# 2. 최종 앙상블 예측 실행
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  All Folds Completed. Starting Final Ensemble Inference..." -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

py src\predict.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✨ Pipeline Finished Successfully! submission.csv is ready." -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Ensemble Inference failed. Please check src\predict.py." -ForegroundColor Red
}

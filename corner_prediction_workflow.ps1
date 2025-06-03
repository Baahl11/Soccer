# Complete workflow for corner prediction system

param (
    [switch]$UseRealData,
    [switch]$SkipDataCollection,
    [switch]$SkipTraining,
    [switch]$SkipEvaluation,
    [switch]$UseFixedEvaluator = $true,
    [int]$DaysToCollect = 180
)

# Function to check if the API key is set
function Check-ApiKey {
    if (-not $env:FOOTBALL_API_KEY -or $env:FOOTBALL_API_KEY -eq "your-api-key-here") {
        Write-Host "API key not set. Please set it first with setup_api_key.ps1" -ForegroundColor Red
        Write-Host "Example: ./setup_api_key.ps1 -ApiKey YOUR_API_KEY" -ForegroundColor Yellow
        Write-Host "If you want to continue with simulated data, use -UseRealData:`$false" -ForegroundColor Yellow
        return $false
    }
    return $true
}

# Create necessary directories if they don't exist
$directories = @("data", "models", "results", "results/plots")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

Write-Host "===== CORNER PREDICTION SYSTEM WORKFLOW =====" -ForegroundColor Cyan
Write-Host ""

# Step 1: Data Collection
if (-not $SkipDataCollection) {
    Write-Host "STEP 1: DATA COLLECTION" -ForegroundColor Cyan

    if ($UseRealData) {
        if (-not (Check-ApiKey)) {
            exit 1
        }
        Write-Host "Collecting real data from API-Football..." -ForegroundColor Yellow
        python corner_data_collector.py --days $DaysToCollect
    } else {
        Write-Host "Generating simulated data for testing..." -ForegroundColor Yellow
        python generate_sample_corners_data.py --samples 1000
    }

    Write-Host "Data collection complete" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "STEP 1: DATA COLLECTION [SKIPPED]" -ForegroundColor Gray
    Write-Host ""
}

# Step 2: Model Training
if (-not $SkipTraining) {
    Write-Host "STEP 2: MODEL TRAINING" -ForegroundColor Cyan
    
    Write-Host "Training Random Forest and XGBoost models..." -ForegroundColor Yellow
    python train_corner_models.py
    
    Write-Host "Model training complete" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "STEP 2: MODEL TRAINING [SKIPPED]" -ForegroundColor Gray
    Write-Host ""
}

# Step 3: Model Evaluation
if (-not $SkipEvaluation) {
    Write-Host "STEP 3: MODEL EVALUATION" -ForegroundColor Cyan
    
    if ($UseFixedEvaluator) {
        Write-Host "Evaluating models with fixed evaluator..." -ForegroundColor Yellow
        python evaluate_corners_model_fixed.py
    } else {
        Write-Host "Evaluating models with standard evaluator..." -ForegroundColor Yellow
        python evaluate_corners_model.py
    }
    
    Write-Host "Model evaluation complete" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "STEP 3: MODEL EVALUATION [SKIPPED]" -ForegroundColor Gray
    Write-Host ""
}

# Step 4: Summary & Next Steps
Write-Host "===== WORKFLOW COMPLETED =====" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "- Data stored in: ./data/" -ForegroundColor White
Write-Host "- Trained models in: ./models/" -ForegroundColor White
Write-Host "- Evaluation results in: ./results/" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Use voting_ensemble_corners.py in your prediction pipeline" -ForegroundColor White
Write-Host "2. Run regular retraining to improve model performance" -ForegroundColor White
Write-Host "3. Consider integrating with your prediction system" -ForegroundColor White
Write-Host ""

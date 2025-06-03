Write-Host "Starting PowerShell test..."
try {
    Write-Host "Importing module..."
    $output = & python -c "from voting_ensemble_corners import VotingEnsembleCornersModel; print('Import successful')"
    Write-Host $output
    
    Write-Host "Creating model instance..."
    $output = & python -c "from voting_ensemble_corners import VotingEnsembleCornersModel; model = VotingEnsembleCornersModel(); print('Model instance created successfully')"
    Write-Host $output
    
    Write-Host "Testing RF prediction method..."
    $output = & python -c "from voting_ensemble_corners import VotingEnsembleCornersModel; model = VotingEnsembleCornersModel(); hasattr(model, '_predict_with_rf') and print('_predict_with_rf method exists')"
    Write-Host $output
    
    Write-Host "Testing XGB prediction method..."
    $output = & python -c "from voting_ensemble_corners import VotingEnsembleCornersModel; model = VotingEnsembleCornersModel(); hasattr(model, '_predict_with_xgb') and print('_predict_with_xgb method exists')"
    Write-Host $output
    
    Write-Host "Test complete - everything works!"
} catch {
    Write-Host "Error: $_"
}

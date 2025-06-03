# Setup API key for soccer prediction models
# This script sets up the API key as an environment variable

param (
    [Parameter(Mandatory=$true)]
    [string]$ApiKey
)

Write-Host "Setting up API key for soccer prediction models..."

# Set environment variable for current session
$env:FOOTBALL_API_KEY = $ApiKey
Write-Host "API key set for current session"

# Create or modify .env file for persistence
$envFile = ".env"
$envContent = Get-Content $envFile -ErrorAction SilentlyContinue
if ($null -eq $envContent) {
    "FOOTBALL_API_KEY=$ApiKey" | Set-Content $envFile
} else {
    $envContent = $envContent | ForEach-Object {
        if ($_ -match "^FOOTBALL_API_KEY=") {
            "FOOTBALL_API_KEY=$ApiKey"
        } else {
            $_
        }
    }
    if ($envContent -notcontains "FOOTBALL_API_KEY=$ApiKey") {
        $envContent += "FOOTBALL_API_KEY=$ApiKey"
    }
    $envContent | Set-Content $envFile
}

Write-Host "API key saved to .env file"
Write-Host ""
Write-Host "To use this API key, run: `$env:FOOTBALL_API_KEY = `"$ApiKey`""
Write-Host "Or restart your terminal after running this script"

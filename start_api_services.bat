@echo off
echo Starting Soccer Prediction API and Structure Fixing Wrapper
echo.
echo Step 1: Starting the main API on port 5000...
start "Soccer Prediction API" cmd /c "cd /D %~dp0 && python app.py"

echo Waiting for main API to initialize (10 seconds)...
timeout /t 10 /nobreak > nul

echo Step 2: Starting the API wrapper on port 5001...
start "Soccer API Structure Wrapper" cmd /c "cd /D %~dp0 && python fixed_api_wrapper.py"

echo.
echo Services have been started!
echo.
echo Main API: http://localhost:5000/api/upcoming_predictions
echo Fixed API: http://localhost:5001/api/fixed_predictions
echo.
echo Use Ctrl+C in each console window to stop the services when done.
echo.

@echo off
REM Start ICU Prediction System

echo ======================================================================
echo ICU MULTI-MODAL PREDICTION SYSTEM - Starting...
echo ======================================================================
echo.

cd /d E:\icu_project

echo [1] Activating conda environment...
call E:\ANACONDA\Scripts\activate.bat icu_project

echo [2] Starting Flask server...
echo.
E:\ANACONDA\envs\icu_project\python.exe app.py

pause

@echo off
cls
echo ===============================================================================
echo ICU Mortality Prediction System - Production Deployment
echo ===============================================================================
echo.
echo Starting the system...
echo.
cd /d "E:\icu_project"
"C:\Users\pande\AppData\Local\Python\pythoncore-3.14-64\python.exe" app_production.py
pause

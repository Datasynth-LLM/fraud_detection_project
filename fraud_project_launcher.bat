@echo off
title Fraud Detection Project Launcher

:menu
cls
echo =============================================
echo        🚀 Fraud Detection Project Menu
echo =============================================
echo 1. Train Model (Run main.py)
echo 2. Launch Streamlit App (Run app.py)
echo 3. Exit
echo =============================================
set /p choice=Enter your choice (1/2/3): 

if "%choice%"=="1" goto train
if "%choice%"=="2" goto app
if "%choice%"=="3" exit

echo Invalid choice. Try again.
pause
goto menu

:train
echo.
echo 🔧 Activating venv and running main.py...
call venv\Scripts\activate
python main.py
pause
goto menu

:app
echo.
echo 🌐 Activating venv and launching Streamlit app...
call venv\Scripts\activate
streamlit run app.py
pause
goto menu

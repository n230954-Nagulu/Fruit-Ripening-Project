@echo off
setlocal
cd /d "%~dp0"

echo.
echo ==========================================
echo   Mango Ripeness Study - Local Launcher
echo ==========================================
echo.

py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11 was not found.
    echo Install Python 3.11 and enable the Python Launcher for Windows.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto :error
)

echo [2/3] Checking and installing project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/3] Starting the Streamlit website...
echo.
echo Open the local URL shown below. Keep this window open while using the app.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error

pause
exit /b 0

:error
echo.
echo [ERROR] Setup or launch failed. Review the error above.
pause
exit /b 1

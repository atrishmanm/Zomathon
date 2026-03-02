@echo off
echo ============================================================
echo     ZOMATO KPT PREDICTION SOLUTION - COMPLETE PIPELINE
echo ============================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install -r requirements.txt

echo.
echo ============================================================
echo Running complete generation pipeline...
echo ============================================================
python run_all.py

echo.
echo ============================================================
echo DONE!
echo ============================================================
echo.
echo Check output/Zomato_KPT_Solution.pdf for the final submission.
echo.
pause

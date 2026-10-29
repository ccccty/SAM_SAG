@echo off
rem SAM_SAG one-click launcher (Windows)
rem Keep messages ASCII-only so the batch file parses safely under any codepage.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   SAM_SAG - Visual Inspection Workstation
echo   (Chinese UI: SAM_SAG ZhiJian ShangWeiJi)
echo ============================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Please create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

echo [1/2] Using Python: %CD%\.venv\Scripts\python.exe
echo [2/2] Starting backend on http://127.0.0.1:8000 ...
echo.

rem Open the browser after a short delay (server needs a moment to boot).
start "" /min cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"

.venv\Scripts\python.exe -m backend.main

echo.
echo Server stopped.
pause
endlocal

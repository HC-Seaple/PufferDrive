@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Drag one or more Waymo JSON files onto this BAT file.
  echo.
  echo Or run:
  echo   waymo_json_to_2d.bat "C:\path\to\scenario.json"
  pause
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

"%PYTHON%" scripts\waymo_json_to_2d.py %* --open
if errorlevel 1 (
  echo.
  echo Conversion failed. Check the message above.
  pause
  exit /b 1
)

echo.
echo Images were saved to: %USERPROFILE%\Downloads\Waymo_2D
pause

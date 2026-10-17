@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.14 -m venv .venv
  if errorlevel 1 exit /b 1
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py --browser.gatherUsageStats false

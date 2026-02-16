@echo off
cd /d "%~dp0"
py -m streamlit run run_ui.py
pause

@echo off
REM ====================================================================
REM run_eval.bat
REM Windows-native runner for Dense-Retrieval LTM Thesis Evaluation
REM
REM Usage:
REM   run_eval.bat                     (5 items, no judge)
REM   run_eval.bat full                (unlimited items, with judge)
REM   run_eval.bat full gemini-2.5-pro (unlimited, custom judge model)
REM ====================================================================

set "PROJ=D:\Downloads\LTMs-in-SLMs"
set "PY=C:\Users\Erika\AppData\Local\Programs\Python\Python312\python.exe"

cd /d "%PROJ%"

REM Load API key from .env
for /f "tokens=1,* delims==" %%a in ('findstr /b "GOOGLE_API_KEY" .env') do set "GOOGLE_API_KEY=%%~b"

set "DATA=Benchmarks\longmemeval_cache\longmemeval_s_cleaned.json"

if /i "%1"=="full" (
    set "MAX="
    set "JUDGE="
    if not "%2"=="" set "JUDGE=--judge-model %2"
) else (
    set "MAX=--max-items 5"
    set "JUDGE=--skip-judge"
)

echo =======================================================================
echo  LTM Thesis Evaluation (Windows)
echo  GPU: NVIDIA GeForce RTX 4050 (6GB)
echo  Data: %DATA%
echo  Items: %MAX:~12% (unlimited if blank)
echo  Judge: %JUDGE%
echo =======================================================================

"%PY%" "Vector-Embedded Memory System\eval\run_combined_eval.py" ^
    --longmemeval-data "%DATA%" ^
    --output-dir "Vector-Embedded Memory System\eval\results" ^
    --quantization 4bit ^
    %MAX% ^
    %JUDGE%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Evaluation failed
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================================
echo  Complete - results in Vector-Embedded Memory System\eval\results\
echo =======================================================================
pause

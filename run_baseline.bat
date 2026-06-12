@echo off
REM ====================================================================
REM run_baseline.bat
REM Windows-native runner for Baseline (No Memory Framework) Evaluation
REM
REM Runs Gemma 3 4B on LongMemEval and LoCoMo datasets with ZERO memory
REM augmentation — the model answers each question from its own knowledge.
REM Stage 2 LLM Judge runs automatically after generation completes.
REM
REM Usage:
REM   run_baseline.bat                         (5 items, auto-skip judge)
REM   run_baseline.bat full                    (all items, auto-run judge)
REM   run_baseline.bat full 4bit 8             (all items, quant, batch-size)
REM   run_baseline.bat full 4bit 8 gemini-2.5-flash  (custom judge model)
REM ====================================================================

set "PROJ=D:\Downloads\LTMs-in-SLMs"
set "PY=C:\Users\Erika\AppData\Local\Programs\Python\Python312\python.exe"

cd /d "%PROJ%"

set "QUANT=4bit"
if not "%2"=="" set "QUANT=%2"

set "BATCH=4"
if not "%3"=="" set "BATCH=%3"

set "JUDGE=gemini-2.5-flash"
if not "%4"=="" set "JUDGE=%4"

set "LME_DATA=Benchmarks\longmemeval_cache\longmemeval_s_cleaned.json"
set "LOCOMO_DATA=Benchmarks\locomo\data\locomo10.json"

if /i "%1"=="full" (
    set "MAX="
    set "JUDGE_FLAGS=--judge-model %JUDGE%"
) else (
    set "MAX=--max-items 5"
    set "JUDGE_FLAGS=--skip-judge"
)

echo =======================================================================
echo  Baseline Evaluation — No Memory Framework
echo  Model: google/gemma-3-4b-it (%QUANT%^)
echo  Batch: %BATCH%
echo  GPU:   NVIDIA GeForce RTX 4050 (6GB^)
echo  LongMemEval: %LME_DATA%
echo  LoCoMo:      %LOCOMO_DATA%
echo  Items: %MAX:~12% (unlimited if blank^)
echo  Judge: %JUDGE_FLAGS%
echo =======================================================================

"%PY%" baseline_eval.py ^
    --longmemeval-data "%LME_DATA%" ^
    --locomo-data "%LOCOMO_DATA%" ^
    --output-dir "results\baseline" ^
    --quantization "%QUANT%" ^
    --batch-size %BATCH% ^
    %MAX% ^
    %JUDGE_FLAGS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Baseline evaluation failed
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================================
echo  Baseline complete - results in results\baseline\RUN_ID\
echo =======================================================================
pause

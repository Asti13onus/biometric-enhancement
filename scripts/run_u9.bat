@echo off
rem U9: minutiae precision + precision-coverage curve on the synthetic test split.
rem Fully resumable (degraded images, GT and enhanced caches all skip-if-exists).
rem ~2.5 h: SNFEN enhancement of 1,290 prints is the long pole. Run from its own
rem window, not inside the IDE.
rem   progress: data\work\u9\run.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\u9\run.log
if not exist data\work\u9 mkdir data\work\u9

for /l %%i in (1,1,5) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\precision_coverage.py >> %LOG% 2>&1
  if not errorlevel 1 goto figure
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:figure
.venv\Scripts\python.exe -u scripts\make_precision_figure.py >> %LOG% 2>&1
echo === finished >> %LOG%

@echo off
rem U12: full benchmark -- methods x held-out sets, wear-severity axis, MINEX position
rem strata. Every cell is one registry row and already-present cells are skipped, so
rem re-running resumes. Several hours; run from its own window, not inside the IDE.
rem   progress: data\work\u12.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\u12.log

for /l %%i in (1,1,6) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\benchmark.py >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

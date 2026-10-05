@echo off
rem Label real, non-test prints with the pyfing teachers (~40 min, 1,160 prints).
rem One process, 4 threads, resumable: re-run this file after any crash to continue.
rem Double-click it, or run it from its own terminal window.
rem   progress: data\work\label_real.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\label_real.log

for /l %%i in (1,1,5) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u scripts\build_real_supervision.py >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

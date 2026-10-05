@echo off
rem U11 final item: does the comparison survive a different extractor (LEADER)?
rem ~50 min (LEADER extracts 640 templates on CPU). Resumable via template caches.
rem Run from its own window.  progress: data\work\paired_leader.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\paired_leader.log

for /l %%i in (1,1,4) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\paired_compare.py --a none --b SNFEN --extractor leader >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

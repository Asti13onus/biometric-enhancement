@echo off
rem Crash-tolerant WAFEN training for this 7.8 GB machine.
rem
rem One process (no DataLoader workers), 4 CPU threads, short epochs so a checkpoint lands
rem every few minutes, and automatic resume from data\work\wafen\checkpoint.pt. If the run
rem dies it is restarted up to 10 times; re-running this file later also just continues.
rem Keep --epochs the same across restarts: the LR schedule is sized from it.
rem
rem   start "" /min scripts\train_simple.bat        (from the repo root)
rem   progress: data\work\train_wafen.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\train_wafen.log

for /l %%i in (1,1,10) do (
  echo === attempt %%i  %date% %time% >> %LOG%
  .venv\Scripts\python.exe -u experiments\train_wafen.py --epochs 10 --max-steps-per-epoch 400 --batch-size 4 --accumulate 4 --workers 0 --threads 4 --limit-val 256 --skip-canary >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished %date% %time% >> %LOG%

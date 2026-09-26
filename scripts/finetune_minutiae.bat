@echo off
rem Fine-tune WAFEN with the minutia-aware loss, from data\work\wafen_ft\best.pt.
rem Writes data\work\wafen_mn\best.pt. ~50 min. One process, auto-resume, retries.
rem Run from its own window (double-click), not inside the IDE.
rem   progress: data\work\finetune_mn.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\finetune_mn.log

for /l %%i in (1,1,10) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\train_wafen.py --real-supervision data\processed\supervision_real --real-repeat 4 --init-from data\work\wafen_ft\best.pt --out-dir data\work\wafen_mn --epochs 8 --max-steps-per-epoch 400 --learning-rate 1e-4 --batch-size 4 --accumulate 4 --workers 0 --threads 4 --skip-canary >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

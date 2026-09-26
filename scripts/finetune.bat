@echo off
rem Fine-tune WAFEN on synthetic Anguli + real non-test prints (teacher distillation).
rem Starts from data\work\wafen\best.pt; writes data\work\wafen_ft\best.pt. ~30 min.
rem Needs scripts\label_real.bat to have finished first.
rem One process, auto-resume, retries on crash. Run it from its own window, not inside
rem the IDE. Keep --epochs the same across restarts: the LR schedule is sized from it.
rem   progress: data\work\finetune.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\finetune.log

for /l %%i in (1,1,10) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\train_wafen.py --real-supervision data\processed\supervision_real --real-repeat 4 --init-from data\work\wafen\best.pt --out-dir data\work\wafen_ft --epochs 8 --max-steps-per-epoch 400 --learning-rate 1e-4 --batch-size 4 --accumulate 4 --workers 0 --threads 4 --skip-canary >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

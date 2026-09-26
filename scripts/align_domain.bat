@echo off
rem Arm two -- post-hoc domain alignment (Joshi line, CDC-style), on the same harness.
rem Starts from the ARM-ONE checkpoint (data\work\wafen\best.pt, synthetic-only); real
rem prints enter undamaged with photometric jitter, labelled by the model itself
rem (scripts\build_pseudo_annotations.py must have run first), with pair-consistency in
rem both domains. Writes data\work\wafen_da\best.pt. ~60 min.
rem One process, auto-resume, retries. Run from its own window, not inside the IDE.
rem   progress: data\work\align_da.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\align_da.log

for /l %%i in (1,1,10) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\train_wafen.py --pairs --real-supervision data\processed\supervision_pseudo --real-mode photometric --real-repeat 4 --init-from data\work\wafen\best.pt --out-dir data\work\wafen_da --epochs 8 --max-steps-per-epoch 400 --learning-rate 1e-4 --batch-size 2 --accumulate 4 --workers 0 --threads 4 --skip-canary >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

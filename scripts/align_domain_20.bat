@echo off
rem Arm two, 20 epochs -- the one pre-registered follow-up. Same recipe as
rem align_domain.bat with only the epoch count changed; separate out-dir so the
rem 8-epoch model stays intact.
rem Starts from the ARM-ONE checkpoint (data\work\wafen\best.pt, synthetic-only); real
rem prints enter undamaged with photometric jitter, labelled by the model itself
rem (scripts\build_pseudo_annotations.py must have run first), with pair-consistency in
rem both domains. Writes data\work\wafen_da20\best.pt. ~2.5 h -- the pre-registered longer run:
rem One process, auto-resume, retries. Run from its own window.
rem   progress: data\work\align_da20.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\align_da20.log

for /l %%i in (1,1,10) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\train_wafen.py --pairs --real-supervision data\processed\supervision_pseudo --real-mode photometric --real-repeat 4 --init-from data\work\wafen\best.pt --out-dir data\work\wafen_da20 --epochs 20 --max-steps-per-epoch 400 --learning-rate 1e-4 --batch-size 2 --accumulate 4 --workers 0 --threads 4 --skip-canary >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

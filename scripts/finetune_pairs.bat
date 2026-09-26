@echo off
rem Pair-consistent fine-tune: two damaged views per print + consistency loss.
rem Starts from data\work\wafen_mn\best.pt; writes data\work\wafen_pc\best.pt. ~70 min.
rem Batch is 2 PAIRS (4 images) -- a pair costs two images of VRAM.
rem One process, auto-resume, retries. Run from its own window, not inside the IDE.
rem   progress: data\work\finetune_pc.log

setlocal
cd /d "%~dp0.."
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=data\work\finetune_pc.log

for /l %%i in (1,1,10) do (
  echo === attempt %%i >> %LOG%
  .venv\Scripts\python.exe -u experiments\train_wafen.py --pairs --real-supervision data\processed\supervision_real --real-repeat 4 --init-from data\work\wafen_mn\best.pt --out-dir data\work\wafen_pc --epochs 8 --max-steps-per-epoch 400 --learning-rate 1e-4 --batch-size 2 --accumulate 4 --workers 0 --threads 4 --skip-canary >> %LOG% 2>&1
  if not errorlevel 1 goto done
  echo attempt %%i failed; resuming in 60 s >> %LOG%
  ping -n 61 127.0.0.1 > nul
)
:done
echo === finished >> %LOG%

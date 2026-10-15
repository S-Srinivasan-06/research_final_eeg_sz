"""Kaggle GPU kernel: stage T3 of path.md, the CatBoost committee (20 base + 20 full + 10 sens models) on TUSZ.

Inputs (private Kaggle datasets; names filled in by scripts/kaggle_push.py):
  <owner>/szfusion-code            this repository's src/ and configs/
  <owner>/szfusion-cbfeats-tusz    {train,dev,eval}/features.parquet + intervals.json (catboost/extract.py)
  this kernel's own earlier output (optional): cb_*.cbm of models already trained, for resuming
The run stops cleanly before starting a new model after TIME_BUDGET_H hours. Push the kernel again to continue: it
resumes from the models already saved.
"""
import glob
import os
import subprocess
import sys
import time

TIME_BUDGET_H = 10.5                  # Kaggle's hard limit is 12 h
WORK = "/kaggle/working"
OUT = f"{WORK}/catboost"


def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        sys.exit(f"[k] not found: {pattern}; inputs: {os.listdir('/kaggle/input')}")
    return hits[0]


for _z in glob.glob("/kaggle/input/**/*.zip", recursive=True):       # dataset folders uploaded in zip dir-mode
    import zipfile
    zipfile.ZipFile(_z).extractall("/kaggle/tmp/unzip/" + os.path.basename(os.path.dirname(_z)))
feat_dir = os.path.dirname(os.path.dirname(find_one("/kaggle/**/train/features.parquet")))
src = os.path.dirname(os.path.dirname(os.path.dirname(find_one("/kaggle/**/szfusion/catboost/train.py"))))  # .../src
cfg = find_one("/kaggle/**/configs/tusz_final.yaml")
prev = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/cb_*.cbm", recursive=True)})
print(f"[k] features={feat_dir}\n[k] src={src}\n[k] resume_from={prev[0] if prev else None}", flush=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost", "optuna", "pyyaml", "pyarrow", "psutil"])
os.makedirs(OUT, exist_ok=True)
cmd = [sys.executable, "-u", "-m", "szfusion.catboost.train", "--config", cfg, "--feat_dir", feat_dir, "--out_dir", OUT,
       "--gpu", "--time_budget_h", str(TIME_BUDGET_H)] + (["--resume_dir", prev[0]] if prev else [])
print("[k] $ " + " ".join(cmd), flush=True)
t0 = time.time()
rc = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=src, PYTHONUNBUFFERED="1")).returncode
print(f"[k] exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
if os.path.exists(f"{OUT}/INCOMPLETE"):
    print("[k] TIME BUDGET REACHED: models saved; push this kernel again to resume", flush=True)

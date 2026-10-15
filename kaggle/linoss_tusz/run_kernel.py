"""Kaggle GPU kernel (2 x T4): stage T4 of path.md, the 5 LinOSS seeds of the final recipe on TUSZ.

Inputs (private Kaggle datasets; their names come from kernel-metadata.json, filled in by scripts/kaggle_push.py):
  <owner>/szfusion-code                 this repository's src/ and configs/
  <owner>/szfusion-tokens-tusz-train    token store, train split  (files named train__tokens.npy, ...)
  <owner>/szfusion-tokens-tusz-dev-eval token store, dev and eval
  <owner>/szfusion-linoss-tusz-state    (optional) this kernel's earlier output, for resuming
Every seed uses the defaults of szfusion.linoss.train (the final recipe). A smoke run is made first. A finished seed
(results.json present in the state) is skipped; an interrupted one resumes exactly from last.pt.
"""
import glob
import os
import shutil
import subprocess
import sys
import threading
import time

SEEDS = {"0": [0, 1, 2], "1": [3, 4]}          # GPU -> seeds
SESSION_HOURS = 11.0
T0 = time.time()
WORK = "/kaggle/working"
os.makedirs(f"{WORK}/runs", exist_ok=True)


def find(pattern, required=True):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits and required:
        sys.exit(f"[k] not found: {pattern}; inputs: {os.listdir('/kaggle/input')}")
    return hits[0] if hits else None


for _z in glob.glob("/kaggle/input/**/*.zip", recursive=True):       # dataset folders uploaded in zip dir-mode
    import zipfile
    zipfile.ZipFile(_z).extractall("/kaggle/tmp/unzip/" + os.path.basename(os.path.dirname(_z)))
store = "/kaggle/tmp/store"                    # flat '<split>__<file>' dataset files -> <split>/<file> links
for f in glob.glob("/kaggle/input/**/*__*", recursive=True):
    sp, name = os.path.basename(f).split("__", 1)
    if sp in ("train", "dev", "eval"):
        os.makedirs(f"{store}/{sp}", exist_ok=True)
        if not os.path.exists(f"{store}/{sp}/{name}"):
            os.symlink(f, f"{store}/{sp}/{name}")
for sp in ("train", "dev", "eval"):
    for name in ("tokens.npy", "y_sec.npy", "index.json"):
        if not os.path.exists(f"{store}/{sp}/{name}"):
            sys.exit(f"[k] token store incomplete: {sp}/{name}")
src = os.path.dirname(os.path.dirname(os.path.dirname(find("/kaggle/**/szfusion/linoss/train.py"))))  # .../src
state = find("/kaggle/input/**/STATE.txt", required=False)
state = os.path.dirname(state) if state else None
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil"])
ENV = dict(os.environ, PYTHONPATH=src, PYTHONUNBUFFERED="1")


def run(name, args, gpu, hours=None):
    out = f"{WORK}/runs/{name}"
    prev = f"{state}/{name}" if state else None
    if prev and os.path.isdir(prev) and not name.startswith("smoke"):
        shutil.copytree(prev, out, dirs_exist_ok=True)
        if os.path.exists(f"{out}/results.json"):
            print(f"[k] {name}: finished in an earlier session", flush=True)
            return 0
    left = hours or SESSION_HOURS - (time.time() - T0) / 3600
    if left < 0.2:
        print(f"[k] {name}: not enough session time left", flush=True)
        return 0
    cmd = [sys.executable, "-u", "-m", "szfusion.linoss.train", "--tokens", store, "--out", out,
           "--max_hours", f"{left:.2f}", *args]
    print(f"[k] GPU {gpu} $ " + " ".join(cmd), flush=True)
    t = time.time()
    with open(f"{WORK}/runs/{name}.stdout.txt", "a") as so:
        rc = subprocess.run(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=gpu), stdout=so, stderr=subprocess.STDOUT).returncode
    print(f"[k] {name}: exit {rc} after {(time.time() - t) / 60:.1f} min", flush=True)
    subprocess.run(f"tail -n 4 {out}/run.log", shell=True)
    return rc


if run("smoke", ["--seed", "1", "--smoke"], "0", hours=0.3) != 0:
    subprocess.run(f"tail -n 30 {WORK}/runs/smoke.stdout.txt", shell=True)
    sys.exit("[k] smoke run failed; not starting the seeds")
shutil.rmtree(f"{WORK}/runs/smoke", ignore_errors=True)
threads = [threading.Thread(target=lambda g=g: [run(f"s{s}", ["--seed", str(s)], g) for s in SEEDS[g]]) for g in SEEDS]
for t in threads:
    t.start()
for t in threads:
    t.join()
with open(f"{WORK}/runs/STATE.txt", "w") as f:
    f.write("szfusion LinOSS TUSZ kernel outputs; publish as szfusion-linoss-tusz-state to resume\n")
print("[k] ALL DONE (or out of session time; push again to continue)", flush=True)

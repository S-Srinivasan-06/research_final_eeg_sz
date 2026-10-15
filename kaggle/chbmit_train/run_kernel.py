"""Kaggle GPU kernel: stage C2 of path.md, CHB-MIT training under the HyBrain protocol. MODE = linoss | catboost.

Inputs: <owner>/szfusion-code, and the outputs of the three prep kernels szfusion-chbmit-prep-{a,b,c} (rec/*.npz,
markers/). Step 1 merges them with szfusion.chbmit_store (z-score fitted on the train clips; clip tables; CatBoost
feature tables). Then:
  linoss    5 seeds of the final recipe in clip-only mode (--clips, L = 12, CHB22 hemisphere swap), 2 × T4
  catboost  the base (20) + sens (10, Optuna) committee on the 588 clip features (configs/chbmit_hybrain.yaml), GPU
Outputs: /kaggle/working/{store_meta,linoss|catboost,catboost_scores}/ (per-second / per-clip dev and test outputs, logs, models).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

MODE = "<MODE>"
SEEDS = {"0": [0, 1, 2], "1": [3, 4]}
T0 = time.time()
W = "/kaggle/working"
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall("/kaggle/tmp/unzip/" + os.path.basename(os.path.dirname(z)))
src = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/**/szfusion/chbmit_store.py", recursive=True))[0]))
cfgdir = os.path.dirname(sorted(glob.glob("/kaggle/**/configs/chbmit_hybrain.yaml", recursive=True))[0])
repo = "/kaggle/tmp/repo"; shutil.rmtree(repo, ignore_errors=True); os.makedirs(repo)
shutil.copytree(src, f"{repo}/src"); shutil.copytree(cfgdir, f"{repo}/configs")
ENV = dict(os.environ, PYTHONPATH=f"{repo}/src", PYTHONUNBUFFERED="1")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "pyarrow", "psutil", "catboost", "optuna",
                "PyWavelets==1.10.0"])
recdirs = sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/rec/*.npz", recursive=True)})
markers = os.path.dirname(sorted(glob.glob("/kaggle/input/**/markers/trainSet_seq2seq_12s_sz.txt", recursive=True))[0])
store = "/kaggle/tmp/chb"
subprocess.run([sys.executable, "-u", "-m", "szfusion.chbmit_store", "--rec", *recdirs, "--markers", markers,
                "--out", store], env=ENV, check=True)
os.makedirs(f"{W}/store_meta", exist_ok=True)
for f in ("stats.json", "clips_train.npz", "clips_dev.npz", "clips_test.npz"):
    shutil.copy(f"{store}/{f}", f"{W}/store_meta/{f}")
shutil.copy(f"{store}/tokens/all/index.json", f"{W}/store_meta/index.json")
shutil.copy(f"{store}/tokens/all/y_sec.npy", f"{W}/store_meta/y_sec.npy")
sigma = json.load(open(f"{store}/stats.json"))["sigma"]
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)


def lin(name, seed, gpu, extra=()):
    out = f"{W}/linoss/{name}"
    cmd = [sys.executable, "-u", "-m", "szfusion.linoss.train", "--tokens", f"{store}/tokens", "--clips", store,
           "--L", "12", "--swap", "chb22", "--gain_std", str(sigma), "--eval_bs", "512", "--out", out, "--seed", str(seed),
           "--max_hours", f"{11.0 - (time.time() - T0) / 3600:.2f}", *extra]
    print(f"[k] GPU {gpu} $ " + " ".join(cmd), flush=True)
    os.makedirs(f"{W}/linoss", exist_ok=True)
    with open(f"{W}/linoss/{name}.stdout.txt", "a") as so:
        rc = subprocess.run(cmd, env=dict(ENV, CUDA_VISIBLE_DEVICES=gpu), stdout=so, stderr=subprocess.STDOUT).returncode
    print(f"[k] {name}: exit {rc} after {(time.time() - T0) / 60:.0f} min of session", flush=True)
    return rc


if MODE == "linoss":
    if lin("smoke", 1, "0", ["--smoke"]) != 0:
        subprocess.run(f"tail -n 30 {W}/linoss/smoke.stdout.txt", shell=True)
        sys.exit("[k] smoke run failed; not starting the seeds")
    shutil.rmtree(f"{W}/linoss/smoke", ignore_errors=True)
    th = [threading.Thread(target=lambda g=g: [lin(f"s{s}", s, g) for s in SEEDS[g]]) for g in SEEDS]
    [t.start() for t in th]; [t.join() for t in th]
elif MODE == "catboost":
    out = f"{W}/catboost"; os.makedirs(out, exist_ok=True)
    cfg = f"{repo}/configs/chbmit_hybrain.yaml"
    subprocess.run([sys.executable, "-u", "-m", "szfusion.catboost.train", "--config", cfg, "--feat_dir",
                    f"{store}/cbfeats", "--out_dir", out, "--gpu", "--time_budget_h", "10.5"], env=ENV, check=True)
    # scores go to their own folder: catboost.train already writes a scores_dev.npz without rec_id/t0 into `out`,
    # which catboost.score would treat as done and skip; eval/chbmit_scores.py keys the clips by rec_id and t0
    subprocess.run([sys.executable, "-u", "-m", "szfusion.catboost.score", "--config", cfg, "--feat_dir",
                    f"{store}/cbfeats", "--models", out, "--out", f"{W}/catboost_scores", "--splits", "dev", "test",
                    "--threads", "4"], env=ENV, check=True)
print("[k] ALL DONE", flush=True)

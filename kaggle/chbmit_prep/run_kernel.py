"""Kaggle CPU kernel: stage C1 of path.md for one group of CHB-MIT patients (PATIENTS).

For each patient:
  1. Download its EDFs and summary from the AWS open-data mirror of PhysioNet (CHB-MIT v1.0.0, ODC-By 1.0;
     https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/).
  2. Run szfusion.tokens_chbmit (tokens, per-second labels, 12 s CatBoost features).
  3. Delete the EDFs.
Also downloads the HyBrain authors' CHB-MIT 12 s split lists at the pinned commit (they are not redistributed in our
repository). Writes versions.json (library versions) and STATE.txt.
Output: /kaggle/working/chb/rec/<recording>.npz, skipped.json, markers/, versions.json.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

MIRROR = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
HYBRAIN = "https://raw.githubusercontent.com/hhyy0401/seizure/694bdd4d9663aef0a5fe846cbdad9f6c979aaaf8/src/data/file_markers_chb"
WORK, RAW = "/kaggle/working/chb", "/kaggle/tmp/chbmit_raw"
patients = "<PATIENTS>".split()                 # filled in by scripts/kaggle_push.py (env PATIENTS)
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall("/kaggle/tmp/unzip/" + os.path.basename(os.path.dirname(z)))
src = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/**/szfusion/tokens_chbmit.py", recursive=True))[0]))
cfgdir = os.path.dirname(sorted(glob.glob("/kaggle/**/configs/tusz_final.yaml", recursive=True))[0])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyedflib==0.1.42", "PyWavelets==1.10.0", "pyyaml"])
os.makedirs(f"{WORK}/markers", exist_ok=True)
for sp in ("train", "dev", "test"):
    for kind in ("sz", "nosz"):
        name = f"{sp}Set_seq2seq_12s_{kind}.txt"
        urllib.request.urlretrieve(f"{HYBRAIN}/{name}", f"{WORK}/markers/{name}")
records = urllib.request.urlopen(f"{MIRROR}/RECORDS").read().decode().split()
# the package reads configs/ relative to the repository root: place configs/ next to src/
repo = "/kaggle/tmp/repo"; shutil.rmtree(repo, ignore_errors=True); os.makedirs(repo)
shutil.copytree(src, f"{repo}/src"); shutil.copytree(cfgdir, f"{repo}/configs")
env = dict(os.environ, PYTHONPATH=f"{repo}/src", PYTHONUNBUFFERED="1")
for p in patients:
    d = f"{RAW}/{p}"; os.makedirs(d, exist_ok=True)
    urllib.request.urlretrieve(f"{MIRROR}/{p}/{p}-summary.txt", f"{d}/{p}-summary.txt")
    for r in [r for r in records if r.startswith(p + "/")]:
        urllib.request.urlretrieve(f"{MIRROR}/{r}", f"{RAW}/{r}")
    print(f"[k] {p}: downloaded {len(os.listdir(d)) - 1} EDFs", flush=True)
    subprocess.run([sys.executable, "-u", "-m", "szfusion.tokens_chbmit", "--raw", RAW, "--out", WORK, "--patients", p,
                    "--workers", "4"], env=env, check=True)
    shutil.copy(f"{d}/{p}-summary.txt", f"{WORK}/{p}-summary.txt")
    shutil.rmtree(d)
import numpy, scipy, pyedflib, pywt  # noqa: E401
json.dump({"numpy": numpy.__version__, "scipy": scipy.__version__, "pyedflib": pyedflib.__version__,
           "pywt": pywt.__version__, "python": sys.version}, open(f"{WORK}/versions.json", "w"))
open(f"{WORK}/STATE.txt", "w").write(f"chbmit prep done for {patients}\n")
print("[k] ALL DONE", flush=True)

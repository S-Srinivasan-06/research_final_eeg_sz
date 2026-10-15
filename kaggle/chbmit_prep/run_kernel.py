"""Kaggle CPU kernel: stage C1 of path.md for one group of CHB-MIT patients (PATIENTS).

For each patient:
  1. Download its EDFs and summary from the AWS open-data mirror of PhysioNet (CHB-MIT v1.0.0, ODC-By 1.0;
     https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/).
  2. Run szfusion.tokens_chbmit (tokens, per-second labels, 12 s CatBoost features).
  3. Delete the EDFs.
Also downloads the HyBrain authors' CHB-MIT 12 s split lists at the pinned commit (they are not redistributed in our
repository). Writes versions.json (library versions) and STATE.txt.
Downloads: object keys are URL-quoted (S3 reads a raw "+" as a space, so chb02/chb02_16+.edf gave HTTP 404 in the
first run of group a), each file is retried with backoff and checked against the mirror's SHA256SUMS.txt.
Workers: a 1 h recording (42 MB EDF) peaks at 1.16 GB per worker (measured on chb01_03), so the worker count of a
patient is capped by the available memory (MEM_PER_EDF_BYTE below) to keep the 4 h recordings of chb04/06/07/09 from
exhausting the 30 GB of a Kaggle CPU session.
Output: /kaggle/working/chb/rec/<recording>.npz, skipped.json, markers/, versions.json.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

MIRROR = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
HYBRAIN = "https://raw.githubusercontent.com/hhyy0401/seizure/694bdd4d9663aef0a5fe846cbdad9f6c979aaaf8/src/data/file_markers_chb"
WORK, RAW = "/kaggle/working/chb", "/kaggle/tmp/chbmit_raw"
patients = "<PATIENTS>".split()                 # filled in by scripts/kaggle_push.py (env PATIENTS)
RETRIES = 6                 # download attempts per file; waits 5, 10, 20, 40, 80 s between attempts
MEM_PER_EDF_BYTE = 35.0     # peak bytes of one worker per EDF byte: 1.16 GB / 42.4 MB = 27.4 measured, + 28 % margin
MAX_WORKERS = 4             # Kaggle CPU session: 4 cores


def fetch(key, dst):
    """Download <MIRROR>/<key> to dst with retries; the key is URL-quoted. Checks SHA256 when the mirror lists one."""
    url = f"{MIRROR}/{urllib.parse.quote(key)}"
    for a in range(RETRIES):
        try:
            urllib.request.urlretrieve(url, dst + ".part")
            if key in sha and hashlib.sha256(open(dst + ".part", "rb").read()).hexdigest() != sha[key]:
                raise IOError(f"SHA256 mismatch for {key}")
            os.replace(dst + ".part", dst)
            return
        except Exception as e:  # noqa: BLE001
            if a == RETRIES - 1:
                raise
            print(f"[k] retry {a + 1} {key}: {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * 2 ** a)


def workers_for(edfs):
    """Worker count that keeps the peak memory of the largest EDFs under 80 % of MemAvailable."""
    avail = next(int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    big = sorted((os.path.getsize(e) for e in edfs), reverse=True)
    n = 0
    while n < min(MAX_WORKERS, len(big)) and sum(big[:n + 1]) * MEM_PER_EDF_BYTE < 0.8 * avail:
        n += 1
    return max(1, n)


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
sha = {}
for line in urllib.request.urlopen(f"{MIRROR}/SHA256SUMS.txt").read().decode().splitlines():
    if line.strip():
        h, k = line.split(maxsplit=1)
        sha[k.strip().lstrip("*")] = h
# the package reads configs/ relative to the repository root: place configs/ next to src/
repo = "/kaggle/tmp/repo"; shutil.rmtree(repo, ignore_errors=True); os.makedirs(repo)
shutil.copytree(src, f"{repo}/src"); shutil.copytree(cfgdir, f"{repo}/configs")
env = dict(os.environ, PYTHONPATH=f"{repo}/src", PYTHONUNBUFFERED="1")
for p in patients:
    d = f"{RAW}/{p}"; os.makedirs(d, exist_ok=True)
    fetch(f"{p}/{p}-summary.txt", f"{d}/{p}-summary.txt")
    recs = [r for r in records if r.startswith(p + "/")]
    for r in recs:
        fetch(r, f"{RAW}/{r}")
    nw = workers_for([f"{RAW}/{r}" for r in recs])
    print(f"[k] {p}: downloaded {len(recs)} EDFs (sha256 checked: {sum(r in sha for r in recs)}), {nw} workers", flush=True)
    subprocess.run([sys.executable, "-u", "-m", "szfusion.tokens_chbmit", "--raw", RAW, "--out", WORK, "--patients", p,
                    "--workers", str(nw)], env=env, check=True)
    shutil.copy(f"{d}/{p}-summary.txt", f"{WORK}/{p}-summary.txt")
    shutil.rmtree(d)
import numpy, scipy, pyedflib, pywt  # noqa: E401
json.dump({"numpy": numpy.__version__, "scipy": scipy.__version__, "pyedflib": pyedflib.__version__,
           "pywt": pywt.__version__, "python": sys.version}, open(f"{WORK}/versions.json", "w"))
open(f"{WORK}/STATE.txt", "w").write(f"chbmit prep done for {patients}\n")
print("[k] ALL DONE", flush=True)

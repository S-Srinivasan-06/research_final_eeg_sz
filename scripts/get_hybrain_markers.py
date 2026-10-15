"""Download the HyBrain authors' split lists (not redistributed here: their repository has no licence).

  python scripts/get_hybrain_markers.py --out <work>/markers

Pinned commit 694bdd4d9663aef0a5fe846cbdad9f6c979aaaf8 of github.com/hhyy0401/seizure:
  tusz/   src/data/file_markers_detection/{devSet,testSet}_seq2seq_12s_{sz,nosz}.txt   (TUSZ 12 s test, units/clips12.py)
  chbmit/ src/data/file_markers_chb/{trainSet,devSet,testSet}_seq2seq_12s_{sz,nosz}.txt (CHB-MIT, chbmit_store.py)
Prints the line count and SHA-256 of every file, so a reader can confirm that their copy matches ours (PROGRESS.md).
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/hhyy0401/seizure/694bdd4d9663aef0a5fe846cbdad9f6c979aaaf8/src/data"
SETS = {"tusz": ("file_markers_detection", ["devSet", "testSet"]),
        "chbmit": ("file_markers_chb", ["trainSet", "devSet", "testSet"])}

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
for name, (folder, splits) in SETS.items():
    d = Path(a.out) / name; d.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        for kind in ("sz", "nosz"):
            f = f"{sp}_seq2seq_12s_{kind}.txt"
            data = urllib.request.urlopen(f"{BASE}/{folder}/{f}").read()
            (d / f).write_bytes(data)
            print(f"{name}/{f}: {len(data.split())} lines  sha256 {hashlib.sha256(data).hexdigest()}")

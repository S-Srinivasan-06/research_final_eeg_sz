"""Stage and upload the private Kaggle datasets of the GPU stages, and push kernels (path.md T3, T4, C2).

  python scripts/kaggle_push.py dataset code                 # src/ + configs/ -> <owner>/szfusion-code
  python scripts/kaggle_push.py dataset tokens-tusz          # token store -> szfusion-tokens-tusz-{train,dev-eval}
  python scripts/kaggle_push.py dataset cbfeats-tusz         # CatBoost features -> szfusion-cbfeats-tusz
  python scripts/kaggle_push.py kernel linoss_tusz|catboost_tusz

The account is the environment variable KAGGLE_OWNER; KAGGLE_CLI names the authenticated Kaggle CLI (default "kaggle"). Datasets are created
private. Every dataset that holds TUSZ-derived data must stay private (TUH data-use agreement; RULES.md 4).
Large files are hard-linked into the staging folder, so staging costs no disk space.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from szfusion import paths  # noqa: E402

OWNER = os.environ.get("KAGGLE_OWNER") or sys.exit("set KAGGLE_OWNER")
CLI = os.environ.get("KAGGLE_CLI", "kaggle").split()          # e.g. "python -m kaggle"
STAGE = paths.work("kaggle_stage", "x").parent


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def upload(slug, folder, title, zip_dirs=False):
    (folder / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": f"{OWNER}/{slug}",
                                                              "licenses": [{"name": "other"}]}))
    exists = subprocess.run(CLI + ["datasets", "status", f"{OWNER}/{slug}"], capture_output=True).returncode == 0
    cmd = (CLI + ["datasets", "version", "-m", "update"] if exists else CLI + ["datasets", "create"]) + \
        ["-p", str(folder)] + (["--dir-mode", "zip"] if zip_dirs else [])
    print("$", " ".join(cmd)); subprocess.run(cmd, check=True)


def dataset(name):
    d = STAGE / name; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    if name == "code":
        shutil.copytree(REPO / "src", d / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "configs", d / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
        upload("szfusion-code", d, "szfusion code", zip_dirs=True)
    elif name == "tokens-tusz":
        root = paths.work("tokens", "tusz")
        for slug, splits in (("szfusion-tokens-tusz-train", ["train"]), ("szfusion-tokens-tusz-dev-eval", ["dev", "eval"])):
            dd = d / slug
            for sp in splits:
                for f in ("tokens.npy", "y_sec.npy", "index.json"):
                    link(root / sp / f, dd / f"{sp}__{f}")
            upload(slug, dd, slug.replace("-", " "))
    elif name == "cbfeats-tusz":
        root = paths.work("cb_feats", "tusz")
        for sp in ("train", "dev", "eval"):
            for f in ("features.parquet", "intervals.json"):
                link(root / sp / f, d / sp / f)
        upload("szfusion-cbfeats-tusz", d, "szfusion cbfeats tusz", zip_dirs=True)


def kernel(name):
    src = REPO / "kaggle" / name; d = STAGE / f"kernel_{name}"; shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d)
    m = d / "kernel-metadata.json"
    txt = m.read_text().replace("<OWNER>", OWNER)
    # token datasets: default = this account's szfusion-tokens-tusz-*; override with any bit-identical store
    txt = txt.replace("<TOKENS_TRAIN>", os.environ.get("TOKENS_TRAIN", f"{OWNER}/szfusion-tokens-tusz-train"))
    txt = txt.replace("<TOKENS_DEV_EVAL>", os.environ.get("TOKENS_DEV_EVAL", f"{OWNER}/szfusion-tokens-tusz-dev-eval"))
    txt = txt.replace("<GROUP>", os.environ.get("GROUP", "a")).replace("<MODE>", os.environ.get("MODE", "linoss"))
    m.write_text(txt)
    k = d / "run_kernel.py"
    k.write_text(k.read_text(encoding="utf-8").replace("<PATIENTS>", os.environ.get("PATIENTS", "")).replace("<MODE>", os.environ.get("MODE", "linoss")), encoding="utf-8")
    subprocess.run(CLI + ["kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    {"dataset": dataset, "kernel": kernel}[sys.argv[1]](sys.argv[2])

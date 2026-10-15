"""Generate the three schematic diagrams of docs/DOCUMENT.md as plain SVG (standard library only, deterministic).

  python src/szfusion/viz/diagrams/make_diagrams.py --out docs/figures            (writes the three .svg files)
  python src/szfusion/viz/diagrams/make_diagrams.py --out docs/figures --check    (also checks the constants below
                                                                                   against the code; needs torch)

Figures:
  dataflow.svg          stages and modules, TUSZ and CHB-MIT (DOCUMENT.md section 1)
  context_timeline.svg  what each branch sees around one second (DOCUMENT.md section 2)
  architecture.svg      the two parallel branches, the stacker and the 1 s / 12 s / 60 s heads (DOCUMENT.md section 3)

No number is drawn that is not a constant of the code. The constants are collected here, each with the place it is
defined; --check imports those places and asserts that the values agree, so a change in the code that is not
reflected here fails loudly.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from xml.sax.saxutils import escape

# ----------------------------------------------------------------------------- constants drawn (source in brackets)
L_WIN, S_WIN = 240, 120          # OscSeq window and final stride, s        [linoss/train.py parse: --L, --final_stride]
CB_WIN, CB_STRIDE = 60, 15       # CatBoost window and stride, s            [configs/tusz_final.yaml catboost.data]
CB_K = 7                         # CatBoost context, windows on each side   [configs/tusz_final.yaml catboost.context.k]
CLIP = 12                        # HyBrain clip, s                          [tokens_chbmit.py CLIP; units/clips12.py]
S_SHORT, S_LONG = 15, 61         # lr_ctx moving averages of z_L, s         [fusion/stacker.py feats]
K_MAX = 61                       # largest 1 s event smoothing, s           [fusion/protocol_1s.py GRID["k"]]
N_FEAT, N_CTX = 588, 7056        # CatBoost base / with-context columns     [catboost/features.py; catboost/context.py]
N_PARAMS = {19: 206_529, 22: 206_625}  # OscSeq parameters                  [linoss/model.py OscSeq, counted]

# colours: LinOSS branch light blue, CatBoost branch light pink, fusion light grey; every outline black
BLUE, PINK, GREY, WHITE, BLACK = "#d6e9f8", "#f9d9e3", "#ececec", "#ffffff", "#000000"
FONT = "Helvetica, Arial, sans-serif"


def weight(t, L=L_WIN):
    """Per-second blending weight inside a window (linoss/data.py RecordingWindows)."""
    return 0.05 + math.sin(math.pi * (t + 0.5) / L) ** 2


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []
        self.warnings = []

    def rect(self, x, y, w, h, fill=WHITE, stroke=BLACK, sw=1.0, dash=None, rx=4):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=12, anchor="start", weight="normal", style="normal", fill=BLACK, width=None):
        if width is not None and 0.5 * size * len(s) > width:
            self.warnings.append(f"text may overflow ({len(s)} chars, {width:.0f} px): {s}")
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
                          f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" fill="{fill}">'
                          f'{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, sw=1.2, dash=None, arrow=False, stroke=BLACK):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                          f'stroke-width="{sw}"{d}{m}/>')

    def path(self, d, sw=1.2, dash=None, arrow=False, fill="none", stroke=BLACK):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arrow)"' if arrow else ""
        self.items.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}{m}/>')

    def box(self, x, y, w, h, lines, fill=WHITE, size=12, bold_first=True, sw=1.0, dash=None):
        """Rectangle with centred lines of text (first line bold)."""
        self.rect(x, y, w, h, fill, sw=sw, dash=dash)
        n = len(lines); lh = size * 1.3
        y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
        for i, s in enumerate(lines):
            self.text(x + w / 2, y0 + i * lh, s, size if i or not bold_first else size, "middle",
                      "bold" if (i == 0 and bold_first) else "normal", width=w - 8)

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>\n'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n')
        Path(path).write_text(head + "\n".join(self.items) + "\n</svg>\n", encoding="utf-8")
        for w in self.warnings:
            print(f"  WARNING {Path(path).name}: {w}")


# ----------------------------------------------------------------------------- figure 1: data flow
def dataflow(out):
    s = SVG(1000, 600)
    s.text(500, 24, "Data flow: stages of path.md section 3 and the modules that run them", 15, "middle", "bold")
    # TUSZ block
    s.rect(10, 38, 980, 300, "none", sw=0.8, dash="5,4")
    s.text(20, 56, "TUSZ v2.0.6 (train / dev / eval)", 12, weight="bold")
    s.box(20, 80, 150, 180, ["TUSZ EDF", ".csv_bi (term)", ".csv (per channel)", "data/tusz.py"], size=11)
    s.box(200, 70, 170, 70, ["T1  tokens.py", "(n_sec, 19, 100) per split", "+ y_sec labels"], BLUE, 11)
    s.box(200, 200, 170, 70, ["T2  catboost/extract.py", "60 s windows every 15 s", f"{N_FEAT} features"], PINK, 11)
    s.box(400, 70, 180, 70, ["T4  linoss/train.py", "OscSeq, seeds 0-4", "prob_{dev,eval} per second"], BLUE, 11)
    s.box(400, 200, 180, 70, ["T3  catboost/train.py, score.py", f"±{CB_K} context ({N_CTX:,} cols)",
                              "50 models, scores per window"], PINK, 11)
    s.box(610, 60, 170, 60, ["T5  fusion/run_1s.py", "stacker + events, dev CV"], GREY, 11)
    s.box(610, 150, 170, 70, ["T6  units/embed.py,", "units/overlooker.py", "12 s and 60 s heads"], GREY, 11)
    s.box(610, 250, 170, 60, ["T7  eval/szcore.py", "timescoring 0.0.7"], WHITE, 11)
    s.box(810, 60, 170, 60, ["1 s test", "every second"], WHITE, 11)
    s.box(810, 150, 170, 70, ["12 s test (HyBrain clips)", "60 s test (Zabihi windows)"], WHITE, 11)
    s.box(810, 250, 170, 60, ["SzCORE events", "results/tusz/szcore.json"], WHITE, 11)
    s.line(170, 105, 198, 105, arrow=True); s.line(170, 235, 198, 235, arrow=True)
    s.line(370, 105, 398, 105, arrow=True); s.line(370, 235, 398, 235, arrow=True)
    s.line(580, 95, 608, 90, arrow=True); s.line(580, 110, 608, 175, arrow=True)
    s.line(580, 225, 608, 105, arrow=True); s.line(580, 240, 608, 195, arrow=True)
    s.line(780, 90, 808, 90, arrow=True); s.line(780, 185, 808, 185, arrow=True)
    s.path("M610,110 L597,110 L597,280 L608,280", arrow=True, dash="3,3")
    s.line(720, 220, 720, 248, arrow=True, dash="3,3"); s.line(780, 280, 808, 280, arrow=True)
    # CHB-MIT block
    s.rect(10, 352, 980, 200, "none", sw=0.8, dash="5,4")
    s.text(20, 370, "CHB-MIT v1.0.0, chb01-chb22, HyBrain within-patient 12 s clip split", 12, weight="bold")
    s.box(20, 385, 150, 70, ["PhysioNet EDF", "chbNN-summary.txt", "data/chbmit.py"], size=11)
    s.box(20, 470, 150, 70, ["HyBrain split lists", "@ 694bdd4d", "{train,dev,test}Set"], size=11)
    s.box(200, 385, 170, 70, ["C1  tokens_chbmit.py", "tokens (n_sec, 22, 100)", "12 s clip features (588)"], WHITE, 11)
    s.box(400, 420, 180, 80, ["C1  chbmit_store.py", "z-score on train clips", "clip tables, label checks"], WHITE, 11)
    s.box(610, 385, 170, 60, ["C2  linoss/train.py --clips", "L = 12, seeds 0-4"], BLUE, 11)
    s.box(610, 470, 170, 70, ["C2  catboost/train.py", "chbmit_hybrain.yaml", "base 20 + sens 10"], PINK, 11)
    s.box(810, 420, 170, 80, ["C3  eval/chbmit_scores.py", "12 s clip + 1 s point-wise", "results/chbmit/*.json"],
          GREY, 11)
    s.line(170, 420, 198, 420, arrow=True); s.line(170, 505, 398, 475, arrow=True)
    s.line(370, 430, 398, 450, arrow=True)
    s.line(580, 450, 608, 415, arrow=True); s.line(580, 470, 608, 500, arrow=True)
    s.line(780, 415, 808, 445, arrow=True); s.line(780, 505, 808, 475, arrow=True)
    # legend
    for i, (c, lab) in enumerate(((BLUE, "LinOSS branch (OscSeq)"), (PINK, "CatBoost branch"),
                                  (GREY, "fusion / heads"), (WHITE, "data, preprocessing, scoring"))):
        x = 20 + 240 * i
        s.rect(x, 568, 22, 14, c, rx=2); s.text(x + 30, 580, lab, 11)
    s.text(980, 596, "dashed arrows: frozen outputs read by the SzCORE scorer", 10, "end", style="italic")
    s.save(Path(out) / "dataflow.svg")


# ----------------------------------------------------------------------------- figure 2: context timeline
def context_timeline(out):
    T_END, T_STAR = 720, 330                           # illustrative recording of 720 s; the second t = 330 s
    s = SVG(1000, 640)
    x0, x1 = 170, 975
    X = lambda t: x0 + (x1 - x0) * t / T_END           # noqa: E731
    s.text(500, 22, "What each branch sees around one second t (offline, non-causal; times in seconds)", 15,
           "middle", "bold")
    # axis
    ya = 600
    s.line(X(0), ya, X(T_END), ya, sw=1)
    for t in range(0, T_END + 1, 60):
        s.line(X(t), ya, X(t), ya + 5, sw=1); s.text(X(t), ya + 18, str(t), 10, "middle")
    s.text(X(T_END), ya + 32, "time since recording start (s), illustrative recording", 10, "end", style="italic")
    for y_a, y_b in ((40, 132), (225, 342), (398, 448), (472, 502), (590, ya)):   # only across the bars
        s.line(X(T_STAR), y_a, X(T_STAR), y_b, sw=1.4, dash="6,4")
    s.text(X(T_STAR) + 4, 50, f"t = {T_STAR} s", 11, weight="bold")

    # A. OscSeq windows
    s.text(10, 78, "OscSeq (TUSZ)", 12, weight="bold")
    s.text(10, 94, f"{L_WIN} s windows", 11); s.text(10, 108, f"every {S_WIN} s", 11)
    s.text(10, 122, "+ end-aligned tail", 11)
    starts = list(range(0, T_END - L_WIN + 1, S_WIN))
    for k, a in enumerate(starts):
        y = 66 + 34 * (k % 2); h = 28
        cover = a <= T_STAR < a + L_WIN
        s.rect(X(a), y, X(a + L_WIN) - X(a), h, BLUE if cover else WHITE, sw=1.6 if cover else 0.8,
               dash=None if cover else "4,3", rx=2)
        pts = " ".join(f"{X(a + t):.1f},{y + h - 3 - (h - 6) * weight(t) / 1.05:.1f}" for t in range(0, L_WIN, 4))
        s.path("M" + pts.replace(" ", " L"), sw=1.0)
    s.text(X(starts[-1] + L_WIN) - 2, 165, "curve: blending weight w(τ) = 0.05 + sin²(π(τ + 0.5)/L), τ = second within the window, L = 240", 10,
           "end", style="italic")
    s.text(X(120), 185, f"p_L(t): weighted mean over the windows covering t (here [120, 360) and [240, 480));",
           10.5)
    s.text(X(120), 199, "inside a window each second sees the whole window, past and future (BiLinOSS)", 10.5)

    # B. CatBoost windows and context
    yb = 225
    s.text(10, yb + 14, "CatBoost (TUSZ)", 12, weight="bold")
    s.text(10, yb + 30, f"{CB_WIN} s windows", 11); s.text(10, yb + 44, f"every {CB_STRIDE} s", 11)
    cov = [a for a in range(0, T_END - CB_WIN + 1, CB_STRIDE) if a <= T_STAR < a + CB_WIN]
    for k, a in enumerate(cov):
        y = yb + 4 + 11 * k
        s.rect(X(a), y, X(a + CB_WIN) - X(a), 9, PINK, sw=1.0, rx=1)
    s.text(X(cov[-1] + CB_WIN) + 6, yb + 26, f"the {len(cov)} windows covering t (starts {cov[0]}-{cov[-1]});",
           10.5)
    s.text(X(cov[-1] + CB_WIN) + 6, yb + 40, "p_C(t) = mean of their scores", 10.5)
    a0 = 300                                                      # context of the window starting at 300 s
    yc = yb + 62
    lo, hi = a0 - CB_K * CB_STRIDE, a0 + CB_K * CB_STRIDE
    for j in range(-CB_K, CB_K + 1):
        a = a0 + j * CB_STRIDE
        s.rect(X(a), yc + 2 + 3.2 * (j + CB_K), X(a + CB_WIN) - X(a), 2.6, PINK if j else BLACK, sw=0.3, rx=0)
    s.line(X(lo), yc + 58, X(hi + CB_WIN), yc + 58, sw=1)
    for xx in (X(lo), X(hi + CB_WIN)):
        s.line(xx, yc + 53, xx, yc + 63, sw=1)
    s.text((X(lo) + X(hi + CB_WIN)) / 2, yc + 75,
           f"±{CB_K}-window context of the window starting at {a0} s: starts {lo}-{hi} (±{CB_K * CB_STRIDE} s), "
           f"signal [{lo}, {hi + CB_WIN}) = {hi + CB_WIN - lo} s", 10.5, "middle")
    s.text(10, yc + 20, f"±{CB_K} windows", 11); s.text(10, yc + 34, "11 statistics", 11)
    s.text(10, yc + 48, "per feature", 11)

    # C. 1 s stacker and event smoothing
    yd = 395
    s.text(10, yd + 14, "1 s stacker", 12, weight="bold"); s.text(10, yd + 30, "lr_ctx and events", 11)
    for k, (w, lab) in enumerate(((S_SHORT, f"S{S_SHORT}(z_L): centred {S_SHORT} s mean"),
                                  (S_LONG, f"S{S_LONG}(z_L): centred {S_LONG} s mean"),
                                  (K_MAX, f"event smoothing k ≤ {K_MAX} s (dev-tuned)"))):
        y = yd + 6 + 16 * k; h = w // 2
        s.rect(X(T_STAR - h), y, X(T_STAR + h + 1) - X(T_STAR - h), 11, GREY, sw=0.8, rx=1)
        s.text(X(T_STAR + h + 1) + 6, y + 10, lab, 10.5)

    # D. CHB-MIT clip-only
    ye = 470
    s.text(10, ye + 14, "CHB-MIT", 12, weight="bold"); s.text(10, ye + 30, "both branches", 11)
    s.text(10, ye + 44, f"{CLIP} s clip only", 11)
    rng = random.Random(0)                                       # illustrative split pattern, not the real lists
    col = {"train": WHITE, "dev": GREY, "test": "#9a9a9a"}
    k_star = T_STAR // CLIP
    for k in range(T_END // CLIP):
        sp = "test" if k == k_star else rng.choices(["train", "dev", "test"], [70, 15, 15])[0]
        s.rect(X(CLIP * k), ye + 6, X(CLIP * (k + 1)) - X(CLIP * k), 22, col[sp], sw=1.6 if k == k_star else 0.5,
               rx=0)
    s.text(X(0), ye + 46, "each clip is scored from its own 12 s; the split is random within patient (70 / 15 / 15),",
           10.5)
    s.text(X(0), ye + 60, "so the neighbours of a test clip are mostly train or dev clips: no statistic may cross "
           "a clip boundary", 10.5)
    for i, (sp, lab) in enumerate((("train", "train clip"), ("dev", "dev clip"), ("test", "test clip"))):
        s.rect(X(0) + 130 * i, ye + 70, 16, 12, col[sp], sw=0.6, rx=0)
        s.text(X(0) + 130 * i + 22, ye + 80, lab + (" (pattern illustrative)" if i == 2 else ""), 10)
    s.save(Path(out) / "context_timeline.svg")


# ----------------------------------------------------------------------------- figure 3: architecture
def architecture(out):
    s = SVG(1000, 900)
    s.text(500, 24, "CatBoost × LinOSS fusion: two parallel branches, the stacker and the 1 s / 12 s / 60 s heads",
           15, "middle", "bold")
    s.box(380, 40, 240, 40, ["one EEG recording", "TUSZ: 19 electrodes; CHB-MIT: 22 bipolar"], WHITE, 11)
    # containers
    s.rect(20, 100, 460, 520, BLUE, sw=1.4)
    s.rect(520, 100, 460, 520, PINK, sw=1.4)
    s.text(250, 122, "LinOSS branch: OscSeq (linoss/model.py)", 13, "middle", "bold")
    s.text(750, 122, "CatBoost branch (catboost/*.py)", 13, "middle", "bold")
    lin = [
        ["tokens (tokens.py): log|FFT| of each 1 s, bins 0-99 Hz, z-scored", "x: (B, L, N, 100)"],
        ["Linear 100 → 32  +  electrode embedding (N × 32)", "(B, L, N, 32)"],
        ["BiLinOSS block × 1, per electrode, weights shared; P = 32", "(B·N, L, 32)"],
        ["[mean ; max] over electrodes  →  Linear 64 → 96", "(B, L, 96)"],
        ["BiLinOSS block × 2 over the whole head; P = 96", "(B, L, 96) = per-second summary h_t"],
        ["head: Linear 96 → 1", "per-second logit ℓ_t: (B, L)"],
        ["blend 240 s windows every 120 s, w(τ) = 0.05 + sin²(π(τ+0.5)/L)", "p_L(t), z_L(t) = logit p_L(t)"],
    ]
    cb = [
        ["16 longitudinal bipolar chains, 200 Hz", "60 Hz notch; 0.5 Hz high-pass (1); 35 Hz low-pass (4)"],
        ["60 s windows, stride 15 s", "(n_windows, 16, 12000)"],
        ["47 features / channel × 4 regions × {median, p95, p95 − median}", f"+ 24 PLV = {N_FEAT} columns"],
        [f"±{CB_K}-window context, 11 statistics per feature", f"{N_FEAT} + 11 × {N_FEAT} = {N_CTX:,} columns"],
        ["committees: base 20 (588) | full 20 (7,056) | sens 10 (7,056, Optuna)", "CatBoost, oblivious trees"],
        ["equal mean of the three committee means", "p_C(window)"],
        ["per second: mean of the 1-4 windows covering it", "p_C(t), z_C(t) = logit p_C(t)"],
    ]
    for col, rows in ((40, lin), (540, cb)):
        for i, r in enumerate(rows):
            y = 138 + 68 * i
            s.box(col, y, 420, 52, r, WHITE, 11.5)
            if i:
                s.line(col + 210, y - 16, col + 210, y - 1, arrow=True)
    s.text(250, 612, f"{N_PARAMS[19]:,} parameters for N = 19; {N_PARAMS[22]:,} for N = 22", 11, "middle",
           style="italic")
    s.line(500, 80, 250, 98, arrow=True); s.line(500, 80, 750, 98, arrow=True)
    # stacker
    s.box(200, 650, 600, 64, ["1 s stacker (fusion/stacker.py), on logits z = logit p",
                              "late: w·Platt(z_C) + (1 − w)·Platt(z_L)  |  lr: [z_C, z_L, z_C·z_L]  |  "
                              "lr_ctx: lr + S15(z_L), S61(z_L)",
                              "chosen by 5-fold patient-grouped CV on dev (out-of-fold AUPRC)"], GREY, 11.5)
    s.line(250, 620, 400, 648, arrow=True); s.line(750, 620, 600, 648, arrow=True)
    # heads
    heads = [
        (40, ["1 s head (fusion/protocol_1s.py)", "p(t) → moving average k s → threshold", "→ merge gaps → min duration",
              "dev rule: max event sens., FA/24 h ≤ 16.56"]),
        (360, ["12 s head (units/overlooker.py)", "overlooker on h_t of the clip:", "LME base + learned correction",
               "WindowStacker [z_C, z_L, z_C·z_L]; best dev F1"]),
        (680, ["60 s head (units/overlooker.py)", "overlooker on h_t of the window,", "WindowStacker on dev;",
               "Zabihi post-processing (windows60.py)"]),
    ]
    for x, r in heads:
        s.box(x, 770, 280, 90, r, GREY, 11)
    s.line(500, 714, 180, 768, arrow=True)
    s.path("M40,436 L10,436 L10,742 L500,742 L500,768", dash="4,3", arrow=True)
    s.path("M500,742 L820,742 L820,768", dash="4,3", arrow=True)
    s.path("M960,504 L992,504 L992,752 L560,752 L560,768", dash="4,3", arrow=True)
    s.path("M900,752 L880,752 L880,768", dash="4,3", arrow=True)
    s.text(14, 736, "h_t of every second (units/embed.py)", 10, style="italic")
    s.text(566, 765, "CatBoost window / clip logit z_C", 10, style="italic")
    s.text(500, 890, "solid: the 1 s path; dashed: inputs of the 12 s and 60 s heads (one LinOSS backbone serves "
           "every unit)", 10, "middle", style="italic")
    s.save(Path(out) / "architecture.svg")


def check():
    """Assert that the constants drawn equal the code's values (imports torch, yaml)."""
    import yaml
    from szfusion import paths
    from szfusion.catboost.extract import load_config
    from szfusion.linoss.model import OscSeq
    from szfusion.linoss.train import parse
    from szfusion.fusion.protocol_1s import GRID
    from szfusion.units.windows60 import STRIDE, WIN
    a = parse(["--tokens", "x", "--out", "y"])
    assert (a.L, a.final_stride) == (L_WIN, S_WIN)
    c = load_config()
    assert (c["data"]["win_sec"], c["data"]["stride_sec"], c["context"]["k"]) == (CB_WIN, CB_STRIDE, CB_K)
    assert (WIN, STRIDE) == (CB_WIN, CB_STRIDE) and max(GRID["k"]) == K_MAX
    for n, p in N_PARAMS.items():
        m = OscSeq(n_el=n, **json_model(a.model))
        assert sum(q.numel() for q in m.parameters()) == p, n
    yaml.safe_load((paths.REPO / "configs" / "chbmit_hybrain.yaml").read_text())
    print("constants agree with the code")


def json_model(s):
    import json
    return json.loads(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.check:
        check()
    for f in (dataflow, context_timeline, architecture):
        f(a.out)
        print(f"wrote {Path(a.out) / (f.__name__ + '.svg')}")

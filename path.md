# path.md: the fixed path of this repository

**Written once on 2026-10-15, before any run in this repository. This file is never edited.** A deviation is allowed only as a labelled `[AMENDMENT]` entry in `PROGRESS.md`. That entry must state what changed, why, and whether the change was decided before or after any test result was seen.

## 1. What this repository reproduces
The **CatBoost × LinOSS fusion**: a seizure detector built from two parallel branches and a logistic stacker.
- The **LinOSS branch** (OscSeq) reads per-second spectral tokens.
- The **CatBoost branch** reads hand-crafted window features.
- The **stacker** fuses the two per-second scores.

The fused score is read out at three resolutions:
- 1 s: every second of a recording;
- 12 s: clips;
- 60 s: windows every 15 s.

**Datasets:**
1. **TUSZ v2.0.6:** train / dev / eval as distributed. Eval is the test set.
2. **CHB-MIT Scalp EEG v1.0.0:** under the HyBrain (Kim et al., CIKM 2026) protocol: chb01–chb22, 12 s non-overlapping clips, within-patient 70/15/15 split given by the HyBrain authors' marker lists. We only compare with their published numbers.

## 2. Directory layout (fixed)
```
path.md            this file (frozen)
PROGRESS.md        append-only labelled log
RULES.md           repository rules (CLAUDE.md points here)
README.md          overview, quick start, official results, data access
configs/           paths.yaml (local data locations; the only place paths live), tusz_final.yaml, chbmit_hybrain.yaml
src/szfusion/      the package
  data/            tusz.py, chbmit.py, montage.py
  tokens.py        per-second tokens
  catboost/        features.py, context.py, train.py, score.py
  linoss/          layer.py, model.py, data.py, train.py
  fusion/          stacker.py, events.py, protocol_1s.py
  units/           embed.py, overlooker.py, clips12.py, windows60.py
  eval/            metrics.py, szcore.py, tables.py
  viz/             evidence.py, diagrams/
scripts/           one command-line entry point per stage (stage numbers below)
kaggle/            kernel templates for the GPU stages
notebooks/         00–12, executed; aggregate outputs only
docs/              DOCUMENT.md (+ DOCUMENT.pdf), figures/
results/           tusz/, chbmit/, dev_run_2026-10-14/  (aggregate JSON/CSV + provenance manifests)
tests/             unit and regression tests
```
Data, tokens, features, model weights trained on TUSZ, per-recording outputs and logs with paths never enter git (see RULES.md). Runtime outputs go to `<work_dir>/` from `configs/paths.yaml`.

## 3. Stages (fixed order, inputs → outputs)

| Stage | Name | Input | Output (under work_dir unless noted) |
|---|---|---|---|
| S0 | port + tests | development code | src/, tests/ pass |
| S1 | exact-reproduction gate | development-run artefacts (local, private) | results/dev_run_2026-10-14/reproduction.json |
| T1 | TUSZ tokens | TUSZ EDF, .csv_bi, .csv | tokens/{train,dev,eval} |
| T2 | CatBoost features | TUSZ EDF | cb_feats/{train,dev,eval} |
| T3 | CatBoost committee | cb_feats/train | catboost/models, scores_{dev,eval} |
| T4 | LinOSS, 5 seeds | tokens | linoss/s{0..4}/best.pt, prob_{dev,eval} |
| T5 | 1 s fusion | T3 + T4 | fusion_1s/stacker.json, dev/eval results |
| T6 | 12 s / 60 s units | T4 embeddings + T3 | units/u12, units/u60 |
| T7 | SzCORE scoring | frozen T5/T6 outputs | results/tusz/szcore.json |
| T8 | agreement table | T5–T7 vs dev run | results/tusz/agreement.json |
| C1 | CHB-MIT data + split lists | PhysioNet + HyBrain markers @ pinned commit | chb/tokens, chb/cb_feats |
| C2 | CHB-MIT training | C1 train clips | chb/linoss/s*, chb/catboost |
| C3 | CHB-MIT fusion + scoring | C2 + dev/test clips | results/chbmit/*.json |
| D1 | figures + document + notebooks | all results | docs/, notebooks/ |
| R1 | release audit | repository | PROGRESS.md [DONE] entry |

## 4. Pre-registered decision rules
1. **Every choice uses dev or train only.** This covers: epochs, stacker form and coefficients, thresholds, event post-processing and overlooker epochs. Test (TUSZ eval, CHB-MIT test) is read only by the scoring step of T5, T6, T7 and C3, once per stage, after the dev selection is written to disk.
2. **LinOSS recipe (fixed):**
   - Oscillator periods are log-uniform in 5–240 s.
   - lr 3e-4, with 300 warm-up steps and a cosine schedule.
   - Weight decay 0.01, except 0 on the LinOSS B and C.
   - EMA 0.998.
   - 6,000 segments of 240 s per epoch, batch 32.
   - At most 30 epochs, patience 6. The best epoch is chosen by dev per-second AUPRC.
   - Seeds 0–4. The ensemble is the mean probability.
3. **CatBoost (fixed, Zabihi et al. 2026):**
   - 16 bipolar chain channels; 60 Hz notch; 0.5 Hz high-pass (order 1); 35 Hz low-pass (order 4).
   - 60 s windows with a 15 s stride; 588 features; ±7-window context gives 7,056 features.
   - Committees:
     - base: 20 models (2 repeats × 10 patient-grouped folds);
     - full: 20 models;
     - sens: 10 models, with Optuna 50 TPE trials (seed 0) on a train split.
   - The score is the equal mean of the three committee means.
4. **1 s fusion:**
   - Candidates are late / lr / lr_ctx.
   - Selection is by 5-fold patient-grouped cross-validation on dev, using out-of-fold AUPRC. Ties within 0.002 go to the simpler candidate.
   - The per-second F1 threshold is the best dev F1.
   - The event rule maximises dev event sensitivity subject to FA/24h ≤ 16.56 (point 1) or ≤ 33.12 (point 2).
   - The grid is k ∈ {1, 5, 15, 31, 61}, gap ∈ {0, 10, 30, 60}, min_dur ∈ {1, 10}, with 40 thresholds at quantiles 0.90–0.9995.
5. **12 s / 60 s:**
   - A residual overlooker is trained on train windows; its epoch is chosen by dev AUPRC (≤ 15 epochs, patience 4).
   - The CatBoost stacker is fitted on dev.
   - 12 s threshold: the best dev F1.
   - 60 s post-processing: the rule of Zabihi et al. (maximise event sensitivity subject to window specificity ≥ 0.93 and FA/h ≤ 0.69).
6. **SzCORE:** timescoring 0.0.7 with its default parameters. It is applied to the frozen outputs, with no re-tuning.
7. **CHB-MIT:**
   - Each 12 s clip is seen alone, by both branches. There is no CatBoost ±7 context.
   - LinOSS uses the same recipe, with the bipolar channels as nodes.
   - The CatBoost features are computed on the 12 s clip.
   - The stacker is [z_C, z_L, z_C·z_L], fitted on dev clips.
   - Threshold: the best dev F1.
   - Reported:
     - 12 s clip AUROC / F1 / AUPRC;
     - 1 s point-wise AUROC / F1;
     - mean ± sd over seeds, and the ensemble.
8. **Acceptance for the from-scratch TUSZ run (T8):**
   - Agreement holds if each official fused metric lies within 2 × the per-seed sd of the development run, measured on the same metric, or within 0.01 when that is larger.
   - A larger difference is reported as `[CHECK]` and explained. It is never corrected by any choice that uses test.
9. **Not adopted (decided on dev before this repository):** test-time augmentation, stride 60, 480 s context at inference, 10 seeds, a gradient-boosted stacker, extra stacker inputs, hysteresis events, Optuna-tuned post-processing, and the 1 s-pooled input to the 12 s / 60 s stackers. They are documented in DOCUMENT.md and are not run here.

## 5. Evidence figures (rules fixed now)
- **Data:** TUSZ eval, from the official run; event point 1.
- **Panel:**
  - 240 s, 19 referential channels, 0.5–35 Hz display filter, a dotted line every 1 s.
  - **Blue:** annotated (per-channel TCP labels mapped to electrodes, OR rule).
  - **Green:** detected (the fused event mask; channel cells where the LinOSS occlusion importance is ≥ 0.5 × the window maximum).
  - **Red:** false alarm.
  - **Teal:** blue and green overlap.
- **Case rules** (no recording IDs are printed):
  - (a) focal-type event of 30–180 s on ≤ 50 % of channels, detected, with the median latency among such events;
  - (b) generalized-type event on all channels, detected, median latency;
  - (c) false alarm of median duration;
  - (d) missed or partial focal-type event of median duration.

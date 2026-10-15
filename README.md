# CatBoost × LinOSS fusion for EEG seizure detection

This is a public reproduction repository for the **CatBoost × LinOSS fusion**, an offline EEG seizure detector. It
has two parallel branches and a logistic stacker:

| Part | What it reads | What it outputs |
|---|---|---|
| **OscSeq**, the LinOSS branch | 1 s log-spectrum tokens | one seizure logit per second, from bidirectional oscillatory state-space layers (LinOSS, Rusch & Rus, ICLR 2025) |
| **CatBoost branch** | 588 hand-crafted features of 60 s windows, with ±7-window context (after Zabihi et al., Sci Rep 2026) | one score per window |
| **Stacker** | both scores, each second | the fused per-second score |

The fused score is read out on three tests:
- every second (**1 s test**);
- 12 s clips (**12 s test**, the HyBrain authors' clips);
- 60 s windows every 15 s (**60 s test**, the windows and event rule of Zabihi et al.).

The datasets and protocols are:
- **TUSZ v2.0.6**: official train / dev / eval splits; eval is the test set.
- **CHB-MIT v1.0.0**: the HyBrain protocol (chb01–chb22, 12 s clips, within-patient split).

Every choice is made on dev or train. Each test set is scored once. The full description is in
[docs/DOCUMENT.md](docs/DOCUMENT.md):
- architecture, formulas and algorithms;
- what "context" means for each branch;
- the protocols, the reproduction guide and the decisions log.

> Status: results are not final. The tables below are placeholders until the official run is complete
> (see `PROGRESS.md`).

## Quick start

```
pip install -r requirements.lock && pip install -e .
cp configs/paths.example.yaml configs/paths.yaml        # set tusz_root, chbmit_root, work_dir
pytest tests
python -m szfusion.tokens build --split eval --out <work_dir>/tokens/tusz --workers 10
```

The other stages, in order, are listed in [docs/DOCUMENT.md §8](docs/DOCUMENT.md#8-reproduction-guide). To
regenerate the diagrams:

```
python src/szfusion/viz/diagrams/make_diagrams.py --out docs/figures [--check]
```

## Data access

- **TUSZ v2.0.6**: register with the Neural Engineering Data Consortium (Temple University) and sign the TUH EEG
  data-use agreement. TUSZ-derived data, weights and per-recording outputs are never committed (RULES.md 4).
- **CHB-MIT v1.0.0**: open on PhysioNet (ODC-By 1.0), physionet.org/content/chbmit/1.0.0/.
- **HyBrain clip lists**: github.com/hhyy0401/seizure at commit `694bdd4d`. They are downloaded, not redistributed.

## Results (TUSZ eval; 5-seed ensembles; settings frozen on dev)

| Test | Model | AUROC | AUPRC | F1 | Events |
|---|---|---|---|---|---|
| 1 s | CatBoost × LinOSS fusion | ⟦RESULT: results/tusz/fusion_1s_eval.json fused_mean⟧ | ⟦…⟧ | ⟦…⟧ | ⟦RESULT: event sens. / FA per 24 h, point 1⟧ |
| 12 s | CatBoost × LinOSS fusion | ⟦RESULT: results/tusz/units12_eval.json look_mean_cb⟧ | ⟦…⟧ | ⟦…⟧ | — |
| 60 s | CatBoost × LinOSS fusion | ⟦RESULT: results/tusz/units60_eval.json look_mean_cb⟧ | ⟦…⟧ | ⟦…⟧ | ⟦RESULT: event sens. / FA per h⟧ |
| SzCORE | CatBoost × LinOSS fusion | — | — | ⟦RESULT: results/tusz/szcore.json event F1⟧ | ⟦RESULT: sens. / FP per day⟧ |

| CHB-MIT (HyBrain protocol) | AUROC | AUPRC | F1 |
|---|---|---|---|
| 12 s clip, fusion | ⟦RESULT: results/chbmit/test_results.json clip12/fused_mean⟧ | ⟦…⟧ | ⟦…⟧ |
| 1 s point-wise, fusion | ⟦RESULT: results/chbmit/test_results.json point1s/fused_mean⟧ | ⟦…⟧ | ⟦…⟧ |

Branch-alone rows, per-seed spreads and the published baselines are in [docs/DOCUMENT.md](docs/DOCUMENT.md)
(section 7.3 and the Appendix).

## Licence

⟦TODO: licence of the code⟧. The datasets keep their own terms: the TUH EEG data-use agreement for TUSZ, and
ODC-By 1.0 for CHB-MIT.

## Citation

⟦TODO: citation of this work⟧. Please also cite LinOSS (Rusch & Rus, ICLR 2025), Zabihi et al. (Sci Rep 16:11283,
2026), HyBrain (Kim et al., CIKM 2026), TUSZ (Shah et al., 2018) and CHB-MIT (Shoeb, 2009; Goldberger et al.,
2000). Full references are in [docs/DOCUMENT.md §10](docs/DOCUMENT.md#10-references).

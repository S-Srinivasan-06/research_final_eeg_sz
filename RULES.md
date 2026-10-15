# Repository rules

1. **`path.md` is frozen.** It is never edited. A deviation is recorded as an `[AMENDMENT]` entry in `PROGRESS.md`, stating what changed, why, and whether any test result had been seen.
2. **Progress is logged, labelled and append-only** in `PROGRESS.md`. Each entry starts with `YYYY-MM-DD HH:MM [LABEL]`. Labels: `[SETUP] [DATA] [RUN] [RESULT] [CHECK] [AMENDMENT] [BLOCKED] [DONE]`. Earlier entries are never rewritten.
3. **Test is read once per stage**, and only by that stage's scoring step, after its dev selection is on disk. Scoring scripts refuse to overwrite an existing test result.
4. **No restricted data in git:**
   - TUSZ signals, tokens or features;
   - per-recording or per-patient outputs;
   - recording or patient IDs;
   - TUSZ-trained weights;
   - logs containing paths;
   - credentials or e-mail addresses.

   Only aggregate results go in `results/`. The release audit greps for all of these.
5. **No absolute paths in code.** Every location comes from `configs/paths.yaml` or an environment variable.
6. **Every number in `README.md` and `docs/` is copied from a file in `results/`** that the code produced. Every result file carries a provenance block: git commit, config hash, date, command.
7. **Every constant is explained where it is defined:** what it is, where it comes from, and its unit.
8. **Pinned dependencies** (`requirements.lock`). New dependencies are logged in `PROGRESS.md`.
9. **Baselines are published values only, cited.** We do not report reproductions of other papers.
10. **Nothing is pushed or published without the maintainer's explicit approval.**

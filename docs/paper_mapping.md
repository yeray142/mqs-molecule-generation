# Paper Mapping: arXiv:2603.22399 → Module Status

Maps sections of arXiv:2603.22399 ("Latent Style-based Quantum Wasserstein GAN
for Drug Design", Baglio/Haddad/Polifka) to this repository's modules.

**A status of DONE here means: implemented, tested, and either verified
against a specific number/formula published in the paper, or confirmed
correct against the paper's own cited source code (MOSES) where the paper
doesn't state enough to verify directly.** Nothing is marked DONE on the
strength of "the file exists" alone — see *Verification methodology* below.

> **Note on a previous version of this file.** An earlier draft of this
> document (generated during initial scaffolding, before implementation
> began) used a fabricated paper structure — "§3 Generator Architecture",
> "§4 Discriminator & Training", "§5 Molecular Generation", "§6
> Experiments", "§7 Conclusion" — none of which exist in the actual paper,
> and listed several modules (`circuits/ansatz.py`, `training/losses.py`,
> `adapters/pennylane_latent_style_adapter.py`, etc.) as DONE that were never
> implemented in this workflow. That draft has been fully replaced by this
> one. Lesson: a paper-section mapping is only as good as its grounding in
> the actual fetched PDF text — it must never be filled in from a generic
> template or a remembered paper structure, and "DONE" must never describe
> planned/stubbed work.

## Real paper structure (verified against the fetched PDF)

| Section | Topic |
|---|---|
| §1 | Introduction |
| §2 | Materials and Methods |
| §2.1 | Dataset construction (MOSES, ZINC Clean Leads, 12,000/4,087 split) |
| §2.2 | Description of the pipeline |
| §2.2.1 | VAE and latent dimension |
| §2.2.2 | Classical and hybrid quantum GAN architecture (discriminator, classical generator, simple/BEL ansätze, Eq. 4 angle map, readout) |
| §2.3 | Metrics (fractions, IntDiv, properties, Wasserstein distances, Eq. 5 significance) |
| §2.4 | Training details |
| §2.4.1 | Training of the VAE |
| §2.4.2 | Latent GAN (WGAN-GP, Adam hyperparameters, n_critic) |
| §3 | Results |
| §3.1 | Classical vs quantum GANs (Table 4) |
| §3.2 | Scenario comparison (Table 5) |
| §3.3 | Inference runs on quantum hardware (Table 6, ibm_kingston) |
| §4 | Discussion and Outlook |
| Supp. §1 | Input distributions (Table 7 — train/test IntDiv, W(X)) |
| Supp. §2–6 | VAE epoch/hyperparameter studies, decoder tests, latent correlations |
| Supp. §7 | QGAN scenario details (Table 14 — Z₀ per metric, ⟨Z₀⟩) |

## Module status

All paths below are relative to `src/mqs_molecule_generation/` unless noted.

| Module | Paper section | Status |
|---|---|---|
| `data/moses.py` | §2.1 | **DONE** — download+checksum, canonicalize, subsample to 12,000/4,087 from MOSES's native `train`/`test_scaffolds` splits, frozen manifest |
| `data/filters.py` + `data/vendor/{mcf,wehi_pains}.csv` | §2.1 ("RDKit filters") | **DONE** — real MOSES `mol_passes_filters` reimplemented from vendored source (not the `molsets` package, which is fragile on modern Python), plus RDKit `FilterCatalog` alternates |
| `metrics/validity.py` | §2.3 (ε_v) | **DONE** |
| `metrics/sa_score.py` | §2.3 (SA) | **DONE** — real RDKit contrib API (`sascorer.calculateScore`), not a class-based API |
| `metrics/diversity.py` | §2.3 (IntDiv) | **DONE** — formula and 1024-bit fingerprint default verified against MOSES's own `internal_diversity`/`fingerprint` source |
| `metrics/properties.py` | §2.3 (LogP, QED, Weight, ε_LogP) | **DONE** |
| `metrics/fractions.py` | §2.3 (ε_d, ε_u, Novelty), Eq. 6 | **DONE** |
| `metrics/wasserstein.py` | §2.3 (W(X)) | **DONE** |
| `metrics/moses_metrics.py` | §2.3 (orchestrator) | **DONE** — `PropertyReport`/`WassersteinReport` |
| `metrics/significance.py` | §2.3, Eq. 5 (Z₀) | **DONE** — reconstructs Table 14 (⟨Z₀⟩ = −1.0129 / −0.3149 vs paper's −1.00 / −0.26) |
| `data/tokenizer.py` | §2.2.1 (SMILES vocabulary) | NOT STARTED |
| `models/vae.py`, `training/vae_trainer.py` | §2.2.1, §2.4.1 | NOT STARTED |
| `models/discriminator.py`, `models/generators/classical.py` | §2.2.2 | NOT STARTED |
| `circuits/{angles,ansatz,simple,bel,readout}.py`, `models/generators/quantum.py` | §2.2.2 | NOT STARTED |
| `training/wgan_gp.py`, GAN training loop | §2.4.2 | NOT STARTED |
| Sampling/decoding pipeline | §2.2, §2.4.2 | NOT STARTED |
| QPUBench adapter | (tooling, not a paper section) | NOT STARTED |
| Hardware inference | §3.3 | NOT STARTED |
| Figures/tables reproduction | §3, Supp. §1–7 | NOT STARTED (the metrics needed to produce them are done; nothing has been run against a trained model yet) |

If any of the NOT STARTED files above already exist elsewhere (e.g. from a
separate Claude Code session outside this workflow), they have **not** been
verified here and should not be treated as done until they go through the
same process as everything above: checked against a specific paper number,
formula, or cited source, with tests to prove it.

## Circuit invariants (target — not yet implemented)

Derived from the paper's N_params table (§2.2.2) and Eq. 4; will be the test
anchors once `circuits/` is built in Phase 3. Not yet coded against — do not
mark `circuits/ansatz.py` DONE without pinning all 11 rows of the source
table exactly as these formulas predict, and without resolving the standing
blocker below.

```
simple: n_angles = n_qb * n_l
bel:    n_angles = n_qb * (5*n_l + 1)      # +1 = final RY per qubit
params  = 2 * n_angles                      # W and b per angle, Eq. 4
D_l     = n_qb (single readout) | 2*n_qb (dual readout)
```

**Standing blocker:** the gate *sequences* for the simple and BEL ansätze
must be read directly from Figures 2 and 3 of the paper — the count formulas
above constrain but do not uniquely determine the gate layout.

## Key findings & decisions (institutional memory)

These are non-obvious, verified findings from implementing §2.1–§2.3. Read
before touching `data/` or `metrics/` again.

- **Filters = 0.734 (Table 3) is unreproduced, and that's expected, not a
  bug.** Neither RDKit's off-the-shelf `FilterCatalog` sets nor the real
  MOSES `mol_passes_filters` (vendored MCF+PAINS, `filter_set: moses_native`)
  reproduce it — `moses_native` lands at exactly 1.0000 on real MOSES data,
  which is correct: MOSES's released dataset was already filtered through
  that exact function when it was built. `moses_native` is pinned anyway,
  since it's the metric MOSES itself defines and the correct one to apply to
  *generated* samples downstream. See `tests/test_data_prep.py::TestAgainstPaper`
  (marked `xfail`, non-blocking).
- **The train-set property means don't match the paper, and the gap predates
  our subsampling.** A fresh 50,000-molecule draw from the real, checksum-
  verified MOSES `train` split (different seed, bypassing our frozen 12k
  entirely) gives mean weight 307.2 and mean QED 0.806 — matching our frozen
  subsample almost exactly (307.2 / 0.805) and confirming the sampling code
  is unbiased. The paper's Table 3 reports 261.5 / 0.596 for the same
  quantity. This rules out a bug in this repo; the likely explanation is an
  undisclosed selection step in the paper's own data preparation. Not
  blocking — recommend contacting the authors. See `scripts/diagnose_pool.py`.
- **`test_scaffolds` vs plain `test`:** switched `valid_split` from MOSES's
  i.i.d. `test` split to `test_scaffolds` (scaffold-disjoint from train, per
  MOSES's own README) partway through investigating Table 7's W(X) gap. This
  measurably improved W(LogP) (58%→15% relative error) but left W(SA)/W(QED)/
  W(Weight) almost unchanged — the opposite of what scaffold-sensitivity
  chemistry would predict if that were the whole story. `test_scaffolds` is
  still the semantically correct choice (matches MOSES's own convention:
  `get_all_metrics`'s docstring says "test (scaffold test)") but is not a
  full explanation for the Table 7 gap on its own.
- **SA score's real API:** `RDConfig.RDContribDir/SA_Score/sascorer.py`
  exposes a module-level `calculateScore(mol)`, not a class. No `SAScore`
  class exists in RDKit's contrib script.
- **IntDiv fingerprint width:** MOSES's own default is 1024-bit Morgan
  fingerprints (`morgan__n=1024` in `moses/metrics/utils.py::fingerprint`),
  not 2048. Formula is the mean over the *full* N×N similarity matrix
  (including the diagonal), confirmed against `average_agg_tanimoto`.
- **ε_LogP range is ambiguous in the paper's own text:** §2.3's prose says
  "between 0 and 5"; Table 3's caption says "|LogP| < 5". Implemented the
  latter (more operational — it describes what was actually computed for the
  published numbers) in `properties.LOGP_RANGE`.
- **Significance ddof (§2.3, Eq. 5):** `MetricStat.from_samples` assumes
  `ddof=1` (sample std) for aggregating 5 seeds. Unverifiable from Table 14
  alone (it only publishes already-computed mean±std, not raw per-seed
  values) — revisit once real 5-seed GAN runs exist to check against.

## Verification methodology

Before marking anything DONE in this table:

1. Find the specific number, formula, or named function in the paper (or, if
   the paper cites MOSES's own definitions without restating them, in
   MOSES's actual source — fetched from `github.com/molecularsets/moses`,
   not recalled from memory).
2. Write a test that reconstructs it. Prefer exact arithmetic on clean inputs
   where possible; where reconstruction is from a rounded published table,
   state the expected reconstruction error and why (see `significance.py`'s
   module docstring for a worked example).
3. Run it against real project data where real data is available (not only
   synthetic fixtures).
4. If a target number can't be reproduced, that's a documented, non-blocking
   finding (see *Key findings* above) — not silence, and not a blind retry
   loop.

## Test coverage

| Test file | Covers |
|---|---|
| `tests/test_data_prep.py` | Split sizes, determinism, canonicalization, filter registry, MOSES-construction filter, download/LFS-pointer handling |
| `tests/test_metrics.py` | Validity, SA score, diversity (incl. analytic formula check), properties, fractions, Wasserstein, `PropertyReport`/`WassersteinReport` integration |
| `tests/test_significance.py` | Exact arithmetic (Z₀ formula, direction map, edge cases), Table 14 reconstruction (per-metric loose tolerance, ⟨Z₀⟩ tight tolerance) |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_data.py` | `calibrate` (filter-set diagnostic vs paper's 0.734) / `prepare` (freeze train/valid splits from config) |
| `scripts/check_metrics.py` | Table 7 gate check (IntDiv, W(X)) against frozen splits, with per-property distribution diagnostics |
| `scripts/diagnose_pool.py` | Full-pool vs frozen-subsample comparison (isolates sampling bugs from upstream data differences) |
| `scripts/check_significance.py` | Table 14 gate check (pure arithmetic, no real data needed) |
# Vendored MOSES filter data

`mcf.csv` and `wehi_pains.csv` are copied verbatim from
[`molecularsets/moses`](https://github.com/molecularsets/moses)
(`moses/metrics/mcf.csv`, `moses/metrics/wehi_pains.csv`), commit as of the
`master` branch on 2026-09-07. MOSES is MIT-licensed:

> Copyright 2018 Insilico Medicine, Inc.

They are vendored rather than pulled in via the `molsets` PyPI package because
that package's pinned dependencies are frequently incompatible with modern
Python/pandas/RDKit. See `mqs_molecule_generation.data.filters.moses_native_passes`
for the reimplementation of `moses.metrics.utils.mol_passes_filters` that
consumes these files.

Do not hand-edit these CSVs; if MOSES updates them upstream, re-fetch from:

- https://raw.githubusercontent.com/molecularsets/moses/master/moses/metrics/mcf.csv
- https://raw.githubusercontent.com/molecularsets/moses/master/moses/metrics/wehi_pains.csv

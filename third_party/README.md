# third_party

Reference repositories cloned for ground truth (2026-10-18). Only the official
ingestion/scoring programs and Dockerfile are tracked; re-clone the rest with:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/aimo-interp/getting-started third_party/getting-started   # de79405 (2026-09-17)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/aimo-interp/baselines       third_party/baselines         # Apache-2.0; data/*.csv
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/aimo-interp/aimo-interp.github.io third_party/website     # website source (2026-10-16)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/byeolki/aimo-interp         third_party/byeolki           # MIT; public participant entry
```

`scripts/run_local.py` executes `getting-started/components/{ingestion,scoring}_program`
verbatim, so local scores use the exact Codabench code path.
`research/import_external.py` copies the public labeled data from these clones into
`data/labeled/` and `data/pool/` (with attribution).

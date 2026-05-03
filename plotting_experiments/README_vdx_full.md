# VDX full preparation

The full VDX breast-cancer microarray dataset is required by `e2e_test.py`.
Both the raw `vdx_dict.npy` (~80 MB) and the converted `vdx_full.npz`
(~22 MB) are gitignored under `plotting_experiments/data/`.

## One-shot setup

Run from the repo root:

```bash
mkdir -p plotting_experiments/data
curl -L -o plotting_experiments/data/vdx_dict.npy \
    https://raw.githubusercontent.com/lorenzomasoero/clustering_replicability/master/real_data/Data/vdx_dict.npy

python -c "
import sys, numpy as np
sys.modules.setdefault('numpy._core', type(sys)('numpy._core'))
sys.modules['numpy._core'].multiarray = np.core.multiarray
sys.modules['numpy._core.multiarray'] = np.core.multiarray
data = np.load('plotting_experiments/data/vdx_dict.npy', allow_pickle=True)
obj = data.item() if data.shape == () else data
df_all = obj['all']
df_pam50 = obj['PAM50']
X_all = df_all.iloc[:, 1:].to_numpy(dtype='float32').T
X_pam50 = df_pam50.iloc[:, 1:].to_numpy(dtype='float32').T
gene_ids_all = df_all.iloc[:, 0].astype(str).to_numpy()
gene_ids_pam50 = df_pam50.iloc[:, 0].astype(str).to_numpy()
sample_ids = np.asarray(df_all.columns[1:], dtype=str)
np.savez_compressed(
    'plotting_experiments/data/vdx_full.npz',
    X_all=X_all, X_pam50=X_pam50,
    gene_ids_all=gene_ids_all, gene_ids_pam50=gene_ids_pam50,
    sample_ids=sample_ids,
)
print('wrote vdx_full.npz')
"
```

## Why a separate `.npz` and not load `.npy` directly

- Upstream `.npy` is a Python dict of pandas DataFrames. `np.load` requires
  `allow_pickle=True` — pickle loading is risky from arbitrary sources.
- The converted `.npz` contains plain `float32` arrays and string IDs, so
  `e2e_test.py` loads it with no special flags.

## Contents of `vdx_full.npz`

| Key              | Shape          | Notes                                    |
|------------------|----------------|------------------------------------------|
| `X_all`          | (344, 22283)   | Full microarray, samples x genes         |
| `X_pam50`        | (344, 90)      | PAM50 marker subset                      |
| `gene_ids_all`   | (22283,)       | Affymetrix probe IDs                     |
| `gene_ids_pam50` | (90,)          | PAM50 probe IDs                          |
| `sample_ids`     | (344,)         | Sample names                             |

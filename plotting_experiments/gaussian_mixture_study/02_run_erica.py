"""Run ERICA pipeline on all 4 Gaussian mixture datasets.

Note: Uses joblib to serialize locally-generated ERICA results.
"""

import os
import sys
import time
import math
import numpy as np
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')

from erica import ERICA

SIGMAS = [0.01, 0.1, 1.0, 10.0]
METHODS = ['kmeans', 'agglomerative_ward', 'agglomerative_single', 'hdbscan']
K_RANGE = [2, 3, 4, 5, 6]
N_ITERATIONS = 200
HDBSCAN_PARAMS = {'min_cluster_size': 15}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('Running ERICA on Gaussian mixture datasets...')
    print(f'  Methods: {METHODS}')
    print(f'  K range: {K_RANGE}, Iterations: {N_ITERATIONS}')
    print()

    for sigma in SIGMAS:
        name = f'gauss4c_sigma{str(sigma).replace(".", "p")}'
        data_path = os.path.join(DATA_DIR, f'{name}.npz')

        if not os.path.exists(data_path):
            print(f'  SKIP {name} (no data file)')
            continue

        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        meta = data['meta'].item()

        print(f'{"="*60}')
        print(f'{name} ({meta["n_samples"]}x{meta["n_features"]}, sigma={sigma})')

        t0 = time.time()
        erica = ERICA(
            data=X,
            k_range=K_RANGE,
            n_iterations=N_ITERATIONS,
            method=METHODS,
            hdbscan_params=HDBSCAN_PARAMS,
            transpose=False,
            output_dir=os.path.join(RESULTS_DIR, 'erica_workdir'),
            verbose=False,
        )
        erica.run()
        er = erica.get_results()
        elapsed = time.time() - t0

        # Summary
        for method in ['kmeans', 'agglomerative_ward', 'agglomerative_single']:
            ks = er.get('k_star', {}).get('TWCRI', {}).get(method, '?')
            if 4 in er['metrics'] and method in er['metrics'][4]:
                m = er['metrics'][4][method]
                twcri = m.get('TWCRI', float('nan'))
                ari = m.get('ARI_mean', float('nan'))
                tw_s = f'{twcri:.3f}' if not math.isnan(twcri) else '  NaN'
                ar_s = f'{ari:.3f}' if not math.isnan(ari) else '  NaN'
            else:
                tw_s, ar_s = '  N/A', '  N/A'
            print(f'  {method:25s} K*={ks:>2}  TWCRI@4={tw_s}  ARI@4={ar_s}')

        hdb = er.get('auto_k', {}).get('hdbscan', {})
        print(f'  {"hdbscan":25s} modal_k={hdb.get("modal_k","?"):>2}  '
              f'agreement={hdb.get("k_agreement_rate",0):.2f}  '
              f'noise_mean={np.mean(hdb.get("noise_counts",[0])):.1f}')
        print(f'  Elapsed: {elapsed:.1f}s')

        # Save
        result = {
            'erica_results': er,
            'auto_k_results': erica.get_auto_k_results('hdbscan'),
            'config': {
                'dataset_name': name,
                'n_samples': meta['n_samples'],
                'n_features': meta['n_features'],
                'true_k': meta['true_k'],
                'sigma': sigma,
                'transpose': False,
            },
        }
        out_path = os.path.join(RESULTS_DIR, f'{name}.joblib')
        joblib.dump(result, out_path)
        print(f'  Saved: {out_path}')
        print()

    print('Gaussian mixture pipeline complete.')


if __name__ == '__main__':
    main()

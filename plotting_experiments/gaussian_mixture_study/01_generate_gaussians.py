"""Generate 4-center 50D Gaussian mixture datasets at varying sigma levels."""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')

N_DIM = 50
N_PER_CLUSTER = 100
K_TRUE = 4
SIGMAS = [0.01, 0.1, 1.0, 10.0]

# Centers: i*3 in dims 0-9, zero in dims 10-49
BASE_CENTERS = np.zeros((4, N_DIM))
for i in range(4):
    BASE_CENTERS[i, :10] = i * 3


def main():
    np.random.seed(42)
    os.makedirs(DATA_DIR, exist_ok=True)
    print('Generating 4-center 50D Gaussian mixtures...')
    print(f'  {N_PER_CLUSTER} samples/cluster, {N_DIM}D, centers spaced by 3 in dims 0-9')
    print(f'  Inter-center Euclidean distance: {3 * np.sqrt(10):.2f}')
    print()

    for sigma in SIGMAS:
        name = f'gauss4c_sigma{str(sigma).replace(".", "p")}'
        Xs = []
        for i in range(K_TRUE):
            Xi = np.random.multivariate_normal(
                BASE_CENTERS[i], np.eye(N_DIM) * sigma**2, N_PER_CLUSTER
            )
            Xs.append(Xi)
        X = np.vstack(Xs)

        meta = {
            'n_samples': X.shape[0],
            'n_features': N_DIM,
            'true_k': K_TRUE,
            'sigma': sigma,
            'description': f'4 Gaussians in {N_DIM}D, sigma={sigma}',
        }
        path = os.path.join(DATA_DIR, f'{name}.npz')
        np.savez(path, X=X, meta=meta)

        expected_radius = sigma * np.sqrt(N_DIM)
        spacing = 3 * np.sqrt(10)
        print(f'  {name}: shape={X.shape}, sigma={sigma}, '
              f'expected_radius={expected_radius:.2f}, spacing={spacing:.2f}, '
              f'ratio={expected_radius/spacing:.2f}')

    print(f'\nDone. Datasets saved to {DATA_DIR}/')


if __name__ == '__main__':
    main()

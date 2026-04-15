#!/usr/bin/env python3
"""12_assignment_profiles.py — Per-cluster assignment scatter & leakage heatmaps.

Port of clam_matrix_sort_18a.m: for each primary cluster, shows how
sample assignments distribute across all other clusters. Reveals
cluster fidelity and boundary ambiguity directly from the CLAM matrix.

Generates per (dataset, method, K):
  - assignment_scatter_{dataset}_{method}_k{K}.png  — one panel per primary cluster
  - assignment_heatmap_{dataset}_{method}_k{K}.png  — K×K mean cross-assignment %
  - assignment_stats_{dataset}_{method}_k{K}.csv    — mean/var per cluster pair
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import set_publication_style, save_figure, METHOD_COLORS, DOUBLE_COL

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'assignment_profiles')

# Gallery datasets + K values to plot. (dataset, K, reason)
TARGETS = [
    # VDX — biological K=3 and TWCRI K*=6
    ('vdx_3gene', 3, 'biological K'),
    ('vdx_3gene', 6, 'TWCRI K*'),
    # Gaussian sigma series — true K=4
    ('gauss4c_sigma0p01', 4, 'true K'),
    ('gauss4c_sigma0p1', 4, 'true K'),
    ('gauss4c_sigma1p0', 4, 'true K'),
    ('gauss4c_sigma10p0', 4, 'true K'),
    # GMM variants
    ('gmm_spherical', 3, 'true K'),
    ('gmm_anisotropic', 3, 'true K'),
    ('gmm_overlapping_pair', 3, 'true K'),
    ('gmm_five_clusters', 5, 'true K'),
]

METHODS = ['kmeans', 'agglomerative_ward']


def normalize_clam(clam):
    """Normalize CLAM rows to assignment proportions."""
    clam = np.array(clam, dtype=float)
    row_sums = clam.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return clam / row_sums


def assignment_profiles(clam):
    """Decompose CLAM by primary cluster assignment.

    Returns dict: primary_cluster -> normalized_rows (n_in_cluster, K)
    """
    normalized = normalize_clam(clam)
    primary = np.argmax(clam, axis=1)
    K = clam.shape[1]
    profiles = {}
    for c in range(K):
        mask = primary == c
        if mask.sum() > 0:
            profiles[c] = normalized[mask]
    return profiles, primary


def plot_scatter(clam, dataset, method, k):
    """Per-cluster assignment scatter — direct port of MATLAB clam_matrix_sort_18a.

    One subplot per primary cluster. Within each subplot, x-axis segments
    show samples' % assignment to each OTHER cluster. Reveals leakage.
    """
    profiles, primary = assignment_profiles(clam)
    K = clam.shape[1]

    fig, axes = plt.subplots(1, K, figsize=(min(DOUBLE_COL * 1.8, 14), 2.8),
                             squeeze=False, sharey=True)
    axes = axes[0]

    for c in range(K):
        ax = axes[c]
        if c not in profiles:
            ax.set_title(f'Cluster {c+1}\n(empty)', fontsize=9)
            ax.set_visible(False)
            continue

        prof = profiles[c]  # (n_samples, K)
        n = prof.shape[0]
        other = [j for j in range(K) if j != c]
        gap = max(int(n * 0.25), 8)

        tick_pos = []
        tick_lbl = []
        offset = 0

        for oc in other:
            x = np.arange(offset, offset + n)
            y = 100.0 * prof[:, oc]
            ax.scatter(x, y, s=1.5, alpha=0.4, color=f'C{oc}', rasterized=True)
            tick_pos.append(offset + n // 2)
            tick_lbl.append(f'C{oc+1}')
            offset += n + gap

        ax.set_title(f'Cluster {c+1}\n(n={n})', fontsize=9)
        ax.set_xlim(-gap, offset)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=7)
        if c == 0:
            ax.set_ylabel('% assigned to other cluster')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylim(-0.5, 55)

    method_label = method.replace('agglomerative_', '').replace('_', ' ').title()
    fig.suptitle(f'{dataset} — {method_label}, K={k}\nPer-cluster assignment profiles',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def plot_heatmap(clam, dataset, method, k):
    """K×K cross-assignment heatmap. Diagonal = fidelity, off-diag = leakage."""
    profiles, _ = assignment_profiles(clam)
    K = clam.shape[1]

    M = np.full((K, K), np.nan)
    for c in range(K):
        if c in profiles:
            M[c] = 100.0 * profiles[c].mean(axis=0)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(M, cmap='YlOrRd', aspect='auto', vmin=0,
                   vmax=max(100, np.nanmax(M)))

    for i in range(K):
        for j in range(K):
            if np.isnan(M[i, j]):
                continue
            color = 'white' if M[i, j] > 50 else 'black'
            ax.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f'C{i+1}' for i in range(K)])
    ax.set_yticklabels([f'C{i+1}' for i in range(K)])
    ax.set_xlabel('Assigned to cluster')
    ax.set_ylabel('Primary cluster')

    method_label = method.replace('agglomerative_', '').replace('_', ' ').title()
    ax.set_title(f'{dataset} — {method_label}, K={k}', fontsize=10)
    plt.colorbar(im, ax=ax, label='Mean %', shrink=0.8)
    fig.tight_layout()
    return fig


def compute_stats(clam, dataset, method, k):
    """Mean and variance of cross-cluster assignments. Returns DataFrame."""
    profiles, _ = assignment_profiles(clam)
    K = clam.shape[1]
    rows = []
    for c in range(K):
        if c not in profiles:
            continue
        prof = profiles[c]
        n = prof.shape[0]
        for j in range(K):
            rows.append({
                'dataset': dataset,
                'method': method,
                'K': k,
                'primary_cluster': c + 1,
                'target_cluster': j + 1,
                'n_samples': n,
                'mean_pct': 100.0 * prof[:, j].mean(),
                'var_pct': (100.0 ** 2) * prof[:, j].var(),
                'std_pct': 100.0 * prof[:, j].std(),
            })
    return pd.DataFrame(rows)


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    all_stats = []

    for dataset, k, reason in TARGETS:
        result_path = os.path.join(RESULTS_DIR, f'{dataset}.joblib')
        if not os.path.exists(result_path):
            print(f'  SKIP {dataset} — no results')
            continue

        data = joblib.load(result_path)
        er = data['erica_results']
        clam_matrices = er['clam_matrices']

        for method in METHODS:
            key = (k, method)
            if key not in clam_matrices:
                print(f'  SKIP {dataset}/{method}/K={k} — not in results')
                continue

            cm = clam_matrices[key]
            tag = f'{dataset}_{method}_k{k}'
            print(f'  {tag} — {cm.shape[0]} samples, K={k} ({reason})')

            # Scatter plot (MATLAB port)
            fig = plot_scatter(cm, dataset, method, k)
            save_figure(fig, f'assignment_scatter_{tag}')

            # Heatmap
            fig = plot_heatmap(cm, dataset, method, k)
            save_figure(fig, f'assignment_heatmap_{tag}')

            # Stats
            df = compute_stats(cm, dataset, method, k)
            csv_path = os.path.join(STATS_DIR, f'assignment_stats_{tag}.csv')
            df.to_csv(csv_path, index=False)
            all_stats.append(df)

    # Combined stats
    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        combined.to_csv(os.path.join(STATS_DIR, 'assignment_stats_all.csv'), index=False)
        print(f'\n  Combined stats: {len(combined)} rows -> assignment_stats_all.csv')

    print('\nDone.')


if __name__ == '__main__':
    main()

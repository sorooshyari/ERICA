# MCSS Algorithm Implementation Notes

*A field guide to the delicate art of asking "but would it cluster the same way twice?"*

This document describes the Monte Carlo Subsampling (MCSS) methodology for evaluating clustering replicability across different clustering algorithms. Because apparently, we can't just cluster data and call it a day—we must also torment ourselves with questions of reproducibility.

## Overview

The MCSS approach tests clustering replicability by:
1. Performing B Monte Carlo iterations with 80-20% train-test splits
2. Recording cluster assignments for left-out (test) data points
3. Aligning cluster identities across iterations (the critical step that keeps us up at night)
4. Building the CLAM matrix to track assignment consistency
5. Computing replicability metrics (CRI, WCRI, TWCRI)

---

## 1. K-means Clustering MCSS

### 1.1 Core Process

**Input:** Dataset of n samples, K range (e.g., K = 2, 3, ..., 7), B iterations

**Output Files:**
| File | Dimensions | Description |
|------|------------|-------------|
| `matrix_data_xxx.csv` | (# test points) × dim | Test point values for iteration xxx |
| `kmeans_predicted_testdata_80_20_1.csv` | B × (# test points) | Cluster assignments per iteration |
| `samples_original_1.csv` | t × dim | Original dataset |
| `clam.csv` | t × K | CLAM matrix (counts per sample per cluster) |

### 1.2 Cluster Identity Alignment (The Part Where sklearn Betrays Us)

**Why it's necessary:** sklearn's K-means, in its infinite wisdom, does not maintain consistent cluster identity across runs. It simply does not care that you called something "Cluster 1" yesterday. The K centroids are arbitrarily sorted into `cluster_centers_xxx.csv` files like socks into a drawer by a particularly apathetic teenager.

**Alignment Algorithm:**
1. Compute L2-norm of cluster centers from K-means
2. Sort centroids from smallest to largest L2 norm
3. For K=4, if sorted indices are [2, 1, 4, 3], then:
   - Original cluster 1 → Global cluster 2
   - Original cluster 2 → Global cluster 1
   - Original cluster 3 → Global cluster 4
   - Original cluster 4 → Global cluster 3
4. Apply mapping to all test point assignments
5. Question your life choices

**Output:** `kmeans_predicted_testdata_80_20_1_aligned.csv`

### 1.3 CLAM Matrix Formation

Run sequentially:
1. `align_cluster_identities_1.r` → aligned predictions
2. `clam_matrix_form_1.r` → B files `CLAM_Siamak_{b}.csv` for b = 0, ..., B-1
3. `sum_to_form_final_CLAM_matrix.r` → `Siamak_formed_CLAM_aligned_clusters.csv`

---

## 2. Agglomerative/Hierarchical Clustering (AC/HC) MCSS

### 2.1 Core Process

Same train-test split approach as K-means, but with an exciting plot twist:
- **Key difference:** AC doesn't provide cluster centers directly. It builds a dendrogram and then looks at you expectantly, as if to say "you figure out where the centers are."
- Must compute centroids by averaging points assigned to each cluster

**Output Files:**
| File | Dimensions | Description |
|------|------------|-------------|
| `matrix_data_xxx.csv` | (# test points) × dim | Test point values |
| `AC_predicted_testdata_80_20_1.csv` | B × (# test points) | Cluster assignments |
| `samples_original_1.csv` | n × dim | Original dataset |

### 2.2 Computing Cluster Centers for AC (DIY Centroid Edition)

Since AC has no native "cluster center" (it's too cool for that), run `compute_ClusterCenters_for_AC_1.r`:

```
For i = 0, 1, ..., B-1:
    For each cluster k in 0, ..., K-1:
        1. Find all test points assigned to cluster k
        2. Compute mean across these points (revolutionary, we know)
        3. Save as centroid for cluster k
    Save centroids to cluster_centers_for_cluster_{i}.csv
```

### 2.3 Alignment Algorithm

Run `align_cluster_identities_2.r`:

```
For each MC iteration:
    a) Compute centroid for each cluster from test points
    b) Take L2 norm of each centroid
    c) Sort K centroids from smallest to largest L2 norm
    d) Assign new indices 0, 1, ..., K-1 based on sorted magnitude
    e) Feel a sense of accomplishment
```

**Output:** `AC_predicted_testdata_80_20_1_aligned.csv`

---

## 3. Affinity Propagation (AP) Clustering MCSS

*"I'll decide how many clusters there should be, thank you very much." — Affinity Propagation*

### 3.1 Special Considerations

**Challenge:** AP does not use a fixed K. Each MC iteration may produce different numbers of clusters {K₁, K₂, ..., K_B}. This is either a feature or a bug, depending on your philosophical outlook.

**Solution:**
1. Record K_i for i = 1, ..., B
2. Compute K^mode (mode of cluster counts)
3. Use histogram/variance/range of {K_i} as "clusterability" measure
4. Only use iterations where K = K^mode for subsequent analysis
5. Pretend the other iterations never happened

### 3.2 Vetting Process

Run `compute_K_for_AP_and_vetFiles_1.r`:
- Counts clusters per iteration
- Generates `vetted_cluster_centers_xxx.csv` for iterations with K^mode clusters
- B^mode = number of iterations with K^mode clusters (the chosen ones)

### 3.3 Alignment

Run `align_cluster_identities_AP_1.r`:
- Same L2-norm sorting approach as K-means
- Only applied to vetted iterations

**Output:** `AP_predicted_testdata_80_20_1_aligned.csv`

**Note:** AP evaluation was marked "STOPPED HERE since AP has issues" — a relatable sentiment for anyone who has worked with AP.

---

## 4. HDBSCAN Clustering MCSS

*The algorithm that decides both how many clusters AND which points are just noise. Bold.*

### 4.1 Parameters

| Parameter | Description | Default | Recommendations |
|-----------|-------------|---------|-----------------|
| `min_cluster_size` | Minimum samples for a cluster | 5 | dim or 2×dim |
| `cluster_selection_epsilon` | Distance threshold for merging | 0.0 | Auto-optimized (it's handling it) |

**Built-in stability:** `cluster_persistence` provides per-cluster stability measure (higher = more stable). HDBSCAN came prepared.

### 4.2 Special Considerations

- HDBSCAN graciously provides both centroids and medoids
- Like AP, may produce different K across iterations (it's an epidemic)
- Must handle variable cluster counts with grace and several R scripts

### 4.3 Processing Pipeline

1. Run `compute_K_for_HDBSCAN_and_vetFiles_1.r` to determine K per iteration
2. Run `align_cluster_identities_HDBSCAN.r`:
   - Compute L2 norm of centroids
   - Sort and assign global indices
3. If multiple K values found, run CLAM formation separately for each K (enjoy)
4. Run `clam_matrix_form_HDBSCAN_1.r`
5. Run `sum_to_form_final_CLAM_matrix.r`

**Output:** `HDBSCAN_predicted_testdata_80_20_1_aligned.csv`

---

## 5. BIRCH Clustering MCSS

*Balanced Iterative Reducing and Clustering using Hierarchies — because someone felt the acronym needed to spell something.*

### 5.1 Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `threshold` (T) | Max radius for subcluster merging | 0.5 | Lower promotes splitting |
| `branching_factor` (B) | Max subclusters per node | 50 | Try 100, 200, 500 for fun |
| `n_clusters` | Final cluster count | 3 | None skips final step (living dangerously) |

**Note:** `subcluster_labels_` and `labels_` contain the same elements but different ordering. Because of course they do.

### 5.2 File Clarification

- `centroids_xxx.csv` — Final cluster centroids (the ones you want)
- `subcluster_centers_xxx.csv` — Subcluster centroids from leaves (not the same thing, don't confuse them, we've all been there)

### 5.3 Processing Pipeline

1. Run `align_cluster_identities_BIRCH.r`:
   - L2 norm sorting of centroids
   - Assign global indices
2. Run `clam_matrix_form_1.r`
3. Run `sum_to_form_final_CLAM_matrix.r`

**Output:** `BIRCH_predicted_testdata_80_20_1_aligned.csv`

---

## 6. Mean Shift Clustering MCSS

### 6.1 Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `bandwidth` (BW) | Kernel bandwidth | None (auto) | Uses `estimate_bandwidth` |

**Open Research Questions (a.k.a. homework):**
- Study `sklearn.cluster.estimate_bandwidth` algorithm — what sorcery does it employ?
- Find rules of thumb for setting BW
- Consider geometric mean of `estimate_bandwidth` and Scott's rule (the former seems too high, the latter too low — a Goldilocks situation)

---

## CLAM Matrix Structure

The CLAM (CLuster Assignment Matrix) is the core data structure, the sacred artifact around which this entire methodology revolves:

```
Dimensions: n_samples × K

Entry M[i, j] = number of times sample i was assigned to cluster j across B iterations

Properties:
- Row sums vary due to random sampling (each sample appears in ~20% of test sets)
- Max entry value = B (if sample was assigned to same cluster every single time — a rare and beautiful occurrence)
- High diagonal concentration = stable clustering
- Low diagonal concentration = existential crisis
```

## Alignment Algorithm Summary

All methods use the same core alignment principle, united in their struggle against sklearn's naming conventions:

```python
def align_clusters(centroids, assignments):
    """
    The Universal Translator for Cluster Identities

    1. Compute L2 norm of each centroid
    2. Sort centroids by L2 norm (smallest to largest)
    3. Create mapping: original_index -> sorted_index
    4. Apply mapping to all assignments
    5. Sleep soundly knowing clusters can now be compared across iterations
    """
    norms = [np.linalg.norm(c) for c in centroids]
    sorted_indices = np.argsort(norms)
    mapping = {orig: new for new, orig in enumerate(sorted_indices)}
    aligned = [mapping[a] for a in assignments]
    return aligned
```

This ensures cluster labels have consistent meaning across MC iterations, restoring order to what would otherwise be chaos.

---

## References

- Parmigiani et al., "Cross-Study Replicability in Cluster Analysis", Statistical Science 2023 — the paper that started it all
- Code repository: https://github.com/lorenzomasoero/clustering_replicability — where the VDX data lives
- Your sanity: location unknown after implementing all of the above

.PHONY: data pipeline plots scatter progressions batch gaussian literature erica-stats all clean

data:
	python 01_generate_datasets.py

pipeline: data
	python 02_run_erica_pipeline.py

plots: pipeline
	python 03_clam_heatmaps.py
	python 04_stability_strips.py
	python 05_metrics_curves.py
	python 06_method_comparison.py
	python 07_surfaces.py

scatter: pipeline
	python 08_cluster_scatter.py

progressions: pipeline
	python 09_k_progression.py
	python 10_full_k_progression.py

batch: pipeline
	python 11_generate_all_plots.py

gaussian: pipeline
	cd gaussian_mixture_study && python 01_generate_gaussian_data.py
	cd gaussian_mixture_study && python 02_run_erica.py
	cd gaussian_mixture_study && python 03_k_progression_grids.py
	cd gaussian_mixture_study && python 04_sigma_curves.py
	cd gaussian_mixture_study && python 05_vdx_comparison.py

literature: pipeline
	cd literature_comparison && python 01_error_bands.py
	cd literature_comparison && python 02_coassignment_heatmap.py
	cd literature_comparison && python 03_pca_erica_scatter.py

erica-stats: pipeline
	cd erica_statistics && python 01_distributions.py
	cd erica_statistics && python 02_statistics_vs_k.py
	cd erica_statistics && python 03_method_comparison.py
	cd erica_statistics && python 04_cross_dataset_summary.py
	cd erica_statistics && python 05_sigma_degradation.py
	cd erica_statistics && python 06_scatter_plots.py
	cd erica_statistics && python 07_pcsp.py
	cd erica_statistics && python 08_ica_analysis.py

all: plots scatter progressions batch gaussian literature erica-stats

clean:
	rm -rf data/ results/ figures/

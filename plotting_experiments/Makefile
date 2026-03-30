.PHONY: data pipeline plots all clean

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

all: plots

clean:
	rm -rf data/ results/ figures/

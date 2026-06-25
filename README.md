# CosmoLensNRE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

Neural Ratio Estimation (NRE) for cosmological parameter inference from strong gravitational lensing images.

## Overview

This repository provides code for the paper:

> **Cosmology Inference from Strong Gravitational Lensing using Neural Ratio Estimation**
> Sreevani Jarugula, Brian Nord, Aleksandra Ćiprijanović, Shubhendu Trivedi

We train a ResNet-based binary classifier on ~2 million simulated DES-noise strong lensing images to learn the likelihood ratio `r(x, θ) = p(x|θ)/p(x)` for the dark energy equation of state `w` and matter density `Ω_m`. Individual per-lens log-ratios are summed across a population of lenses for joint cosmological inference. We demonstrate rank-histogram-based post-hoc calibration and show the method is ~100× more data-efficient than analytical likelihood approaches.

## Pipeline

*[Figure placeholder: NRE pipeline diagram — `fig:nre_pipeline`]*

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/deepskies/CosmoLensNRE.git
cd CosmoLensNRE
conda env create -f environment.yml
conda activate <env-name>
```

Or with Poetry (Python ≥ 3.10):

```bash
pip install poetry
poetry install
```

## Repository Structure

```
CosmoLensNRE/
├── notebooks/
│   ├── train_2M.ipynb                          # Train NRE classifier on 2M lensing images
│   ├── compare_analytical_nre_likelihood.ipynb # Compare NRE vs. analytical Einstein-radius likelihood
│   ├── plot_dataset.ipynb                      # Visualize training images
│   ├── plot_image_posterior.ipynb              # Per-lens posteriors: uncalibrated vs. calibrated
│   ├── plot_parity_residuals_regions.ipynb     # Parity plots and residuals across parameter regions
│   ├── plot_population_posteriors.ipynb        # Population-level (w, Ω_m) posteriors
│   ├── visualize_grid.ipynb                    # Test data regions in (Ω_m, w) parameter space
│   └── w0_om0_degeneracy.ipynb                 # w–Ω_m degeneracy in Einstein radius
├── src/scripts/
│   ├── evaluate_with_calibration.py            # MCMC sampling with calibrated log-ratio
│   └── evaluate_without_calibration.py         # MCMC sampling with uncalibrated NRE log-ratio
├── environment.yml
├── pyproject.toml
└── LICENSE.txt
```

## Data

Lensing images are simulated using [deeplenstronomy](https://github.com/deepskies/deeplenstronomy) with DES-like noise on a 32×32 pixel grid. Each image is flux-normalized so that pixels sum to `edge_size²`. The training set contains ~2 million images spanning a grid of `(w, Ω_m)` values.
The train, test, and calibrated dataset is available on Zenodo.

## Authors

- Sreevani Jarugula (Fermilab)
- Brian Nord (Fermilab / University of Chicago)
- Aleksandra Ćiprijanović (Fermilab)
- Shubhendu Trivedi (University of Chicago)

## Citation

If you use this code, please cite the paper:



## License

This project is licensed under the MIT License — see [LICENSE.txt](LICENSE.txt) for details.

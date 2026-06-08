# A Controlled Benchmark of Quantum-Latent GAN Augmentation for Brain MRI

Code and protocol for the controlled, parameter-matched comparison of a
quantum-latent generator against a classical generator for brain-MRI
augmentation. The pipeline encodes images into a KL-regularized VAE latent
space, trains a conditional WGAN-GP with either a variational quantum
generator or a parameter-matched classical generator, decodes synthetic
samples, and evaluates downstream classification across labeled-data fractions
with multiple seeds and paired statistics.

## Dataset
Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor), Mendeley
Data V1, 2025, doi:10.17632/zwr4ntf94j.1 (a re-release of the public figshare,
SARTAJ, and Br35H collections). Download it and arrange as an `ImageFolder`:

```
<DATA_ROOT>/Train/<class>/*.jpg
<DATA_ROOT>/Test/<class>/*.jpg
```

Point the code at it with an environment variable:

```bash
export BRAIN_MRI_DATA=/path/to/brain_tumor_mri_dataset
```

## Requirements
Python 3.10+, and:

```bash
pip install torch torchvision pennylane torchmetrics torch-fidelity \
            scikit-learn scikit-image scipy matplotlib tqdm pandas
```

A GPU is strongly recommended. The main script auto-installs the metric
backends (`torchmetrics`, `scikit-image`, `torch-fidelity`) if missing.

## Files
- `qlqgan_pipeline.py` — main experiment. Runs the VAE -> WGAN-GP (quantum and
  classical) -> ResNet-18 pipeline over data fractions {5, 10, 25, 100}% with
  8 classifier seeds, and writes `raw_results.csv`, `summary.csv`, sample
  grids, and t-SNE plots to `qlqgan_results/`. Intra-set diversity (mean
  pairwise SSIM, pixel std) is computed at every fraction.
- `fill_placeholders.py` — appendix experiments: generator-initialization
  variance (retraining generators over seeds) and latent maximum mean
  discrepancy (MMD) between real and synthetic codes.
- `mmd_from_scratch.py` — standalone NumPy MMD (RBF kernel, unbiased estimator,
  permutation test); no external MMD dependency.
- `make_figs.py` — rebuilds the figures used in the paper from the saved
  results/images.

## Reproducing the main results
```bash
export BRAIN_MRI_DATA=/path/to/brain_tumor_mri_dataset
python qlqgan_pipeline.py
```

Key settings (top of `qlqgan_pipeline.py`): `FRACTIONS`, `SEEDS`,
`VAE_EPOCHS`, `GAN_EPOCHS`, `CLS_EPOCHS`, `N_QUBITS`, `Q_DEPTH`. Set
`QUICK_MODE = True` for a fast end-to-end sanity run.

## Seed protocol
The VAE and both generators are trained once per data fraction (fixed
`GEN_SEED`); synthetic sets are generated once and reused across the eight
classifier seeds, so the variance reported in the main tables reflects
classifier training. Generator-initialization variance is measured separately
in `fill_placeholders.py`.

## Note
Quantum circuits run on an ideal state-vector simulator (PennyLane
`default.qubit`); results are not subject to hardware noise.

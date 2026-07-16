# Sketch-to-Face Synthesis Using a Conditional GAN

A pix2pix-style conditional GAN (U-Net generator + conditional PatchGAN discriminator)
that translates hand-drawn face sketches into photorealistic face images, trained on the
[Person Face Sketches](https://www.kaggle.com/datasets/almightyj/person-face-sketches)
dataset.

Built for Generative AI — Assignment 2, Question 3.

## Contents

- [`q3_conditional_gan.ipynb`](q3_conditional_gan.ipynb) — full training/inference
  notebook (dataset loading, U-Net generator, conditional PatchGAN discriminator,
  training loop with mixed precision and checkpoint resume, evaluation, inference).
- [`report.tex`](report.tex) — IEEE conference-format technical report (LaTeX source,
  Overleaf-ready). [`report_overleaf.zip`](report_overleaf.zip) bundles it with the
  figures needed to compile directly.
- [`loss_curve.png`](loss_curve.png) — generator/discriminator training loss over 100
  epochs.
- [`training_history.json`](training_history.json) / [`run_summary.json`](run_summary.json)
  — raw per-epoch loss and timing data from the training run.
- [`zip_results.py`](zip_results.py) — utility to bundle sample outputs and training
  history for download from a Kaggle session.
- [`requirements.txt`](requirements.txt) — Python dependencies.

## Method

- **Generator:** U-Net with 8 stride-2 encoder blocks and 7 decoder blocks, skip
  connections between matching-resolution encoder/decoder layers.
- **Discriminator:** Conditional 70×70 PatchGAN — classifies overlapping patches of the
  (sketch, photo) pair as real/fake rather than the whole image.
- **Loss:** Adversarial loss + λ=100 weighted L1 reconstruction loss (pix2pix objective).
- Trained for 100 epochs (6,000 subsampled training pairs, 1,000 validation pairs) on a
  single Kaggle GPU session, using automatic mixed precision and cuDNN autotuning to fit
  the time budget.

## Results

| Metric | Value |
|---|---|
| Final generator loss | 31.58 |
| Final discriminator loss | 0.266 |
| Validation L1 ([-1,1] scale) | 0.3035 |
| Validation PSNR | 13.30 dB |
| Total training time | 111.7 min (~1.86 h) |

See [`report.tex`](report.tex) for the full writeup, architecture diagrams, and analysis
of these results.

## Usage

Open [`q3_conditional_gan.ipynb`](q3_conditional_gan.ipynb) in Kaggle or Colab with a
GPU runtime. The notebook auto-locates the dataset (Kaggle "Add Data" mount or
`kagglehub` download), trains the model, and saves checkpoints after every epoch for
resumable training.

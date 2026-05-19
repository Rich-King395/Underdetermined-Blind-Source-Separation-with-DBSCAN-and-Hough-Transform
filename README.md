# DBSCAN-Hough UBSS Mixing Matrix Estimation

This folder reproduces the mixing-matrix estimation pipeline from:

Sun et al., "Novel mixing matrix estimation approach in underdetermined blind source separation", Neurocomputing, 2016.

The implementation focuses on the paper's two-observation underdetermined case:

```text
x(t) = A s(t),  m = 2, n = 3
```

## Method

The reproduction follows the paper's estimation route:

1. Transform observations into the time-frequency domain using STFT.
2. Detect single-source TF points by phase consistency.
3. Remove low-energy TF points.
4. Map retained points onto the upper unit hypersphere.
5. Use DBSCAN to estimate the number of source signals.
6. Apply Hough transform inside each DBSCAN cluster to refine the line direction.
7. Use the refined line directions as columns of the estimated mixing matrix.

## Dataset

The synthetic dataset is generated in `data/`:

```text
data/S_synthetic.npy   source signals, shape (3, 40000)
data/A_synthetic.npy   mixing matrix, shape (2, 3)
data/X_synthetic.npy   observations, shape (2, 40000)
data/fs_synthetic.npy  sampling rate
```

The sources overlap strongly in time, so the time-domain observation scatter is not a clean set of lines. Their frequency bands are separated, so STFT improves sparsity and makes single-source TF point detection meaningful.

Regenerate the dataset:

```bash
python generate_data.py
```

## Run

Estimate the mixing matrix:

```bash
python main.py
```

Use the paper-style phase-ratio detector:

```bash
python main.py --phase-rule ratio
```

Useful parameters:

```bash
python main.py --epsilon 0.06 --energy-fraction 0.1 --min-samples 10
python main.py --nperseg 512 --noverlap 384 --beta 180
```

Create a visualization:

```bash
python visualize.py --no-show
```

The default output is:

```text
figures/dbscan_hough_synthetic.png
```

## Files

```text
generate_data.py   creates the synthetic UBSS dataset
dbscan_hough.py    STFT, single-source detection, DBSCAN, Hough refinement
main.py            command-line reproduction entry point
visualize.py       plots time-domain and TF-domain distributions
README.md          this file
```

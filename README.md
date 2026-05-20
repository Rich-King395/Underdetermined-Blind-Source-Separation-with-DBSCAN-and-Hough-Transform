# DBSCAN-Hough UBSS Mixing Matrix Estimation

This folder reproduces the mixing-matrix estimation and source-recovery
pipeline inspired by:

Sun et al., "Novel mixing matrix estimation approach in underdetermined blind source separation", Neurocomputing, 2016.

The implementation focuses on the paper's two-observation underdetermined case:

```text
x(t) = A s(t),  m = 2, n = 3
```

## Method

The mixing-matrix estimation follows the paper's route:

1. Transform observations into the time-frequency domain using STFT.
2. Detect single-source TF points by phase consistency.
3. Remove low-energy TF points.
4. Map retained points onto the upper unit hypersphere.
5. Use DBSCAN to estimate the number of source signals.
6. Apply Hough transform inside each DBSCAN cluster to refine the line direction.
7. Use the refined line directions as columns of the estimated mixing matrix.

After estimating `A_hat`, the implementation performs source recovery in the
STFT domain:

1. For each TF observation vector, enumerate candidate active-source submatrices.
2. Solve least squares with each candidate submatrix.
3. Select the candidate with the minimum reconstruction residual.
4. Fill the estimated source STFT coefficients.
5. Apply ISTFT to reconstruct time-domain source signals.

For the default two-observation underdetermined case, the default active source
count is `m - 1 = 1`. This matches the single-source dominant TF sparsity
assumption; choosing two active sources with only two observations can make many
candidate submatrices fit with nearly zero residual.

## Dataset

The synthetic dataset is generated in `data/`:

```text
data/S_synthetic.npy   source signals, shape (3, 40000)
data/A_synthetic.npy   mixing matrix, shape (2, 3)
data/X_synthetic.npy   noisy observations, shape (2, 40000)
data/X_clean_synthetic.npy   clean observations before environmental noise
data/fs_synthetic.npy  sampling rate
data/snr_synthetic.npy observation SNR in dB
```

The sources overlap strongly in time and their frequency bands are partially
overlapped. Environmental noise is added to the observations, so the dataset is
less ideal than a perfectly TF-separated sparse mixture while still retaining
enough STFT-domain structure for the DBSCAN-Hough pipeline.
The default observation SNR is 30 dB.

Regenerate the dataset:

```bash
python generate_data.py
python generate_data.py --snr-db 15
```

## Run

Most default dataset, STFT, DBSCAN, Hough, recovery, and visualization
parameters are centralized in `parameter.py`. Edit that file when you want to
change the project-wide defaults; command-line arguments can still override
them for one run.

Estimate the mixing matrix and recover sources:

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
python main.py --active-count 1
python main.py --min-cluster-size 50
python main.py --min-cluster-fraction 0.15
```

Create a visualization:

```bash
python visualize.py --no-show
```

The default output is:

```text
figures/dbscan_hough_synthetic.png
figures/dbscan_hough_synthetic_signals.png
figures/dbscan_hough_synthetic_dbscan.png
```

## Files

```text
parameter.py      central editable parameter configuration
generate_data.py   creates the synthetic UBSS dataset
dbscan_hough.py    STFT, phase-threshold SSP detection, DBSCAN, Hough, recovery
main.py            command-line reproduction entry point
visualize.py       plots distributions, DBSCAN clusters, and separated signals
README.md          this file
```

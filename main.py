import argparse
from pathlib import Path

import numpy as np

import dbscan_hough
from generate_data import save_dataset


def load_dataset(dataset="synthetic", data_dir="data"):
    data_dir = Path(data_dir)
    source_path = data_dir / f"S_{dataset}.npy"
    mixing_path = data_dir / f"A_{dataset}.npy"
    observation_path = data_dir / f"X_{dataset}.npy"
    fs_path = data_dir / f"fs_{dataset}.npy"

    if not source_path.exists():
        if dataset != "synthetic":
            raise FileNotFoundError(f"Dataset {dataset!r} was not found in {data_dir}.")
        save_dataset(data_dir)

    S = np.load(source_path)
    A = np.load(mixing_path)
    X = np.load(observation_path)
    fs = int(np.load(fs_path))
    return S, A, X, fs


def normalized_nmse(A, A_hat, pairs):
    if not pairs:
        return np.nan

    A_matched = np.column_stack([A[:, source_index] for source_index, _ in pairs])
    A_hat_matched = np.column_stack([A_hat[:, estimated_index] for _, estimated_index in pairs])
    return 10 * np.log10(
        np.sum((A_hat_matched - A_matched) ** 2) / (np.sum(A_matched**2) + 1e-12)
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce Sun et al. 2016 DBSCAN-Hough mixing matrix estimation."
    )
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--noverlap", type=int, default=384)
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--energy-fraction", type=float, default=0.1)
    parser.add_argument("--phase-rule", choices=["angle", "ratio"], default="angle")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--beta", type=int, default=180)
    parser.add_argument("--rho-tolerance", type=float, default=0.015)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    S, A, X, fs = load_dataset(args.dataset, args.data_dir)

    A_hat, diagnostics = dbscan_hough.estimate_mixing_matrix(
        X,
        fs=fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        epsilon=args.epsilon,
        energy_fraction=args.energy_fraction,
        phase_rule=args.phase_rule,
        min_samples=args.min_samples,
        eps=args.eps,
        beta=args.beta,
        rho_tolerance=args.rho_tolerance,
    )

    pairs, angles = dbscan_hough.match_columns_by_angle(A, A_hat)

    print("\nDataset:", args.dataset)
    print("Source shape:", S.shape)
    print("Observation shape:", X.shape)
    print("Sampling rate:", fs)
    print("STFT shape:", diagnostics["stft_shape"])
    print("Detected single-source TF points:", diagnostics["single_source_count"])
    print("DBSCAN Eps:", diagnostics["eps"])
    print("DBSCAN MinPts:", diagnostics["min_samples"])
    print("Estimated source count:", diagnostics["cluster_count"])
    print("Cluster sizes:", diagnostics["cluster_sizes"])
    print("Noise points:", diagnostics["noise_count"])

    print("\nTrue mixing matrix A:\n", A)
    print("\nEstimated mixing matrix A_hat:\n", A_hat)
    print("\nColumn matches (true -> estimated):", pairs)
    print("Deviation angles in degrees:", np.round(angles, 6))
    print("Mean deviation angle:", float(np.mean(angles)) if len(angles) else np.nan)
    print("NMSE(dB):", normalized_nmse(A, A_hat, pairs))

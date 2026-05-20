import argparse
from pathlib import Path

import numpy as np

import dbscan_hough
import parameter
from generate_data import save_dataset


def load_dataset(dataset=parameter.DATASET_NAME, data_dir=parameter.DATA_DIR):
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


def best_scaled_source_metrics(S, S_hat):
    used = set()
    pairs = []
    correlations = []
    mse_values = []

    for source_index, source in enumerate(S):
        best = None
        for recovered_index, recovered in enumerate(S_hat):
            if recovered_index in used:
                continue

            scale = np.dot(source, recovered) / (np.dot(recovered, recovered) + 1e-12)
            aligned = scale * recovered
            mse = np.mean((source - aligned) ** 2)
            corr = np.corrcoef(source, aligned)[0, 1]
            score = abs(corr)
            if best is None or score > best[0]:
                best = (score, recovered_index, scale, mse, corr)

        if best is not None:
            _, recovered_index, scale, mse, corr = best
            used.add(recovered_index)
            pairs.append((source_index, recovered_index, scale))
            mse_values.append(mse)
            correlations.append(corr)

    return pairs, np.asarray(correlations), np.asarray(mse_values)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce Sun et al. 2016 DBSCAN-Hough mixing matrix estimation."
    )
    parser.add_argument("--dataset", default=parameter.DATASET_NAME)
    parser.add_argument("--data-dir", default=parameter.DATA_DIR)
    parser.add_argument("--nperseg", type=int, default=parameter.STFT_NPERSEG)
    parser.add_argument("--noverlap", type=int, default=parameter.STFT_NOVERLAP)
    parser.add_argument("--epsilon", type=float, default=parameter.PHASE_EPSILON)
    parser.add_argument("--energy-fraction", type=float, default=parameter.ENERGY_FRACTION)
    parser.add_argument("--phase-rule", choices=["angle", "ratio"], default=parameter.PHASE_RULE)
    parser.add_argument("--min-samples", type=int, default=parameter.DBSCAN_MIN_SAMPLES)
    parser.add_argument("--min-cluster-size", type=int, default=parameter.MIN_CLUSTER_SIZE)
    parser.add_argument("--min-cluster-fraction", type=float, default=parameter.MIN_CLUSTER_FRACTION)
    parser.add_argument("--eps", type=float, default=parameter.DBSCAN_EPS)
    parser.add_argument("--beta", type=int, default=parameter.HOUGH_BETA)
    parser.add_argument("--rho-tolerance", type=float, default=parameter.HOUGH_RHO_TOLERANCE)
    parser.add_argument(
        "--active-count",
        type=int,
        default=parameter.ACTIVE_COUNT,
        help="Number of active sources per TF bin for minimum-residual recovery. Default is m - 1.",
    )
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
        min_cluster_size=args.min_cluster_size,
        min_cluster_fraction=args.min_cluster_fraction,
        eps=args.eps,
        beta=args.beta,
        rho_tolerance=args.rho_tolerance,
    )

    pairs, angles = dbscan_hough.match_columns_by_angle(A, A_hat)
    S_hat, recovery_diagnostics = dbscan_hough.recover_sources_min_residual(
        X,
        A_hat,
        fs=fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        active_count=args.active_count,
    )
    source_pairs, source_corrs, source_mse = best_scaled_source_metrics(S, S_hat)

    print("\nDataset:", args.dataset)
    print("Source shape:", S.shape)
    print("Observation shape:", X.shape)
    print("Sampling rate:", fs)
    print("STFT shape:", diagnostics["stft_shape"])
    print("Phase rule:", args.phase_rule)
    print("Detected single-source TF points:", diagnostics["single_source_count"])
    print("DBSCAN Eps:", diagnostics["eps"])
    print("DBSCAN MinPts:", diagnostics["min_samples"])
    print("Minimum cluster size:", diagnostics["min_cluster_size"])
    print("Estimated source count:", diagnostics["cluster_count"])
    print("Cluster sizes:", diagnostics["cluster_sizes"])
    print("Noise points:", diagnostics["noise_count"])

    print("\nTrue mixing matrix A:\n", A)
    print("\nEstimated mixing matrix A_hat:\n", A_hat)
    print("\nColumn matches (true -> estimated):", pairs)
    print("Deviation angles in degrees:", np.round(angles, 6))
    print("Mean deviation angle:", float(np.mean(angles)) if len(angles) else np.nan)
    print("NMSE(dB):", normalized_nmse(A, A_hat, pairs))

    print("\nRecovered source shape:", S_hat.shape)
    print("Recovery active source count:", recovery_diagnostics["active_count"])
    print("TF active-combination counts:", recovery_diagnostics["assignment_counts"])
    print("Source matches (true -> recovered, scale):", source_pairs)
    print("Recovered source correlations:", np.round(source_corrs, 6))
    print("Best scaled source MSE:", float(np.mean(source_mse)) if len(source_mse) else np.nan)

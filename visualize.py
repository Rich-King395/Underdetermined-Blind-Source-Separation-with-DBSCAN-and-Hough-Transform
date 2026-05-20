import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dbscan_hough
import parameter
from main import best_scaled_source_metrics, load_dataset


def _draw_directions(ax, A, color, label_prefix):
    for index in range(A.shape[1]):
        vector = A[:, index] / (np.linalg.norm(A[:, index]) + 1e-12)
        ax.plot(
            [0, vector[0]],
            [0, vector[1]],
            color=color,
            linewidth=2,
            label=f"{label_prefix} {index + 1}" if index == 0 else None,
        )


def _stft_pair_scatter_values(Z, magnitude_threshold=0.0):
    pair_coefficients = Z[:2].reshape(2, -1)
    magnitudes = np.linalg.norm(pair_coefficients, axis=0)
    pair_coefficients = pair_coefficients[:, magnitudes > magnitude_threshold]

    x_values = np.concatenate([pair_coefficients[0].real, pair_coefficients[0].imag])
    y_values = np.concatenate([pair_coefficients[1].real, pair_coefficients[1].imag])
    return x_values, y_values


def _set_symmetric_limits(ax, x_values, y_values):
    limit = 1.05 * max(
        np.max(np.abs(x_values)) if len(x_values) else 0.0,
        np.max(np.abs(y_values)) if len(y_values) else 0.0,
        1e-12,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)


def _plot_signal_group(axes, time, signals, title, prefix, sample_count):
    for index, signal in enumerate(signals):
        axes[index].plot(time[:sample_count], signal[:sample_count], linewidth=0.9)
        axes[index].set_ylabel(f"{prefix}{index + 1}")
        axes[index].grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")


def build_signal_figure(S, X, S_hat, fs, sample_count=4000):
    pairs, correlations, mse_values = best_scaled_source_metrics(S, S_hat)
    S_hat_display = np.zeros_like(S)
    for source_index, recovered_index, scale in pairs:
        S_hat_display[source_index] = scale * S_hat[recovered_index]

    source_count = S.shape[0]
    observation_count = X.shape[0]
    recovered_count = S_hat_display.shape[0]
    row_count = max(source_count, observation_count, recovered_count)
    time = np.arange(max(S.shape[1], X.shape[1], S_hat_display.shape[1])) / fs

    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(14, 2.1 * row_count),
        sharex=True,
        constrained_layout=True,
    )
    if row_count == 1:
        axes = axes.reshape(1, 3)

    for column in range(3):
        for row in range(row_count):
            axes[row, column].axis("off")

    _plot_signal_group(axes[:source_count, 0], time, S, "Source signals", "s", sample_count)
    _plot_signal_group(axes[:observation_count, 1], time, X, "Observed mixtures", "x", sample_count)
    _plot_signal_group(
        axes[:recovered_count, 2],
        time,
        S_hat_display,
        "Recovered signals (matched/scaled)",
        "r",
        sample_count,
    )

    for column in range(3):
        for row in range(row_count):
            if axes[row, column].has_data():
                axes[row, column].axis("on")

    fig.suptitle(
        "Blind source separation result: "
        f"mean |corr|={np.mean(np.abs(correlations)):.4f}, "
        f"mean scaled MSE={np.mean(mse_values):.4e}",
        fontsize=13,
    )
    return fig


def build_dbscan_figure(A, A_hat, diagnostics):
    points = diagnostics["points"]
    labels = diagnostics["labels"]
    centers = diagnostics["dbscan_centers"]
    cluster_labels = sorted(label for label in np.unique(labels) if label != dbscan_hough.NOISE_LABEL)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    axes[0].scatter(points[:, 0], points[:, 1], s=5, alpha=0.35, color="0.55", edgecolors="none")
    _draw_directions(axes[0], A, color="tab:red", label_prefix="true")
    axes[0].set_title("DBSCAN input: detected single-source TF points")
    axes[0].set_xlabel("|X1(t, f)| normalized")
    axes[0].set_ylabel("|X2(t, f)| normalized")

    for color_index, label in enumerate(cluster_labels):
        cluster_points = points[labels == label]
        color = cmap(color_index % 10)
        axes[1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=8,
            alpha=0.75,
            color=color,
            edgecolors="none",
            label=f"cluster {label} ({len(cluster_points)})",
        )

    if np.any(labels == dbscan_hough.NOISE_LABEL):
        noise = points[labels == dbscan_hough.NOISE_LABEL]
        axes[1].scatter(
            noise[:, 0],
            noise[:, 1],
            s=10,
            alpha=0.45,
            color="0.3",
            marker="x",
            label=f"noise ({len(noise)})",
        )

    if len(centers):
        axes[1].scatter(
            centers[:, 0],
            centers[:, 1],
            s=80,
            color="white",
            edgecolors="black",
            linewidths=1.2,
            marker="o",
            label="DBSCAN mean centers",
        )
    _draw_directions(axes[1], A_hat, color="black", label_prefix="Hough")
    axes[1].set_title("DBSCAN clusters and Hough-refined directions")
    axes[1].set_xlabel("|X1(t, f)| normalized")
    axes[1].set_ylabel("|X2(t, f)| normalized")
    axes[1].legend(loc="lower left", fontsize=8)

    for ax in axes:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "DBSCAN clustering visualization: "
        f"Eps={diagnostics['eps']:.4f}, MinPts={diagnostics['min_samples']}, "
        f"min_cluster_size={diagnostics['min_cluster_size']}",
        fontsize=13,
    )
    return fig


def build_figure(S, A, X, fs, args):
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
    S_hat, recovery_diagnostics = dbscan_hough.recover_sources_min_residual(
        X,
        A_hat,
        fs=fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        active_count=args.active_count,
    )
    Z = dbscan_hough.compute_stft_matrix(
        X,
        fs=fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
    )
    all_tf_points = dbscan_hough.complex_tf_vectors_to_real_points(
        Z.reshape(Z.shape[0], -1)
    )
    selected_points = diagnostics["points"]
    labels = diagnostics["labels"]
    freq_x, freq_y = _stft_pair_scatter_values(
        Z,
        magnitude_threshold=args.tf_scatter_threshold,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    time = np.arange(S.shape[1]) / fs
    for source_index, source in enumerate(S):
        axes[0, 0].plot(
            time[: args.source_preview_samples],
            source[: args.source_preview_samples],
            linewidth=0.9,
            label=f"s{source_index + 1}",
        )
    axes[0, 0].set_title("Source signals (time-overlapped)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].scatter(X[0], X[1], s=2, alpha=0.25, edgecolors="none")
    _draw_directions(axes[0, 1], A, color="tab:red", label_prefix="true")
    axes[0, 1].set_title("Observation scatter in time domain")
    axes[0, 1].set_xlabel("x1(t)")
    axes[0, 1].set_ylabel("x2(t)")
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].grid(True, alpha=0.25)
    _set_symmetric_limits(axes[0, 1], X[0], X[1])

    axes[0, 2].scatter(freq_x, freq_y, s=2, alpha=0.18, edgecolors="none")
    axes[0, 2].set_title("Observation scatter in STFT domain")
    axes[0, 2].set_xlabel("Re/Im STFT(x1)")
    axes[0, 2].set_ylabel("Re/Im STFT(x2)")
    axes[0, 2].set_aspect("equal", adjustable="box")
    axes[0, 2].grid(True, alpha=0.25)
    _set_symmetric_limits(axes[0, 2], freq_x, freq_y)

    axes[1, 0].scatter(all_tf_points[:, 0], all_tf_points[:, 1], s=2, alpha=0.08, edgecolors="none")
    _draw_directions(axes[1, 0], A, color="tab:red", label_prefix="true")
    axes[1, 0].set_title("All STFT magnitude directions")
    axes[1, 0].set_xlabel("|X1(t, f)| normalized")
    axes[1, 0].set_ylabel("|X2(t, f)| normalized")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].scatter(selected_points[:, 0], selected_points[:, 1], s=5, alpha=0.4, edgecolors="none")
    _draw_directions(axes[1, 1], A, color="tab:red", label_prefix="true")
    axes[1, 1].set_title("Detected single-source TF directions")
    axes[1, 1].set_xlabel("|X1(t, f)| normalized")
    axes[1, 1].set_ylabel("|X2(t, f)| normalized")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].grid(True, alpha=0.25)

    cluster_labels = sorted(label for label in np.unique(labels) if label != dbscan_hough.NOISE_LABEL)
    _draw_directions(axes[1, 2], A_hat, color="black", label_prefix="estimated")
    for label in cluster_labels:
        cluster_points = selected_points[labels == label]
        axes[1, 2].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=8,
            alpha=0.75,
            edgecolors="none",
            label=f"cluster {label}",
        )
    if np.any(labels == dbscan_hough.NOISE_LABEL):
        noise = selected_points[labels == dbscan_hough.NOISE_LABEL]
        axes[1, 2].scatter(noise[:, 0], noise[:, 1], s=5, alpha=0.25, color="0.6", label="noise")
    axes[1, 2].set_title("Single-source TF points after DBSCAN-Hough")
    axes[1, 2].set_xlabel("|X1(t, f)| normalized")
    axes[1, 2].set_ylabel("|X2(t, f)| normalized")
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].grid(True, alpha=0.25)

    for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"DBSCAN-Hough reproduction: {diagnostics['cluster_count']} sources estimated, "
        f"{diagnostics['single_source_count']} single-source TF points, "
        f"active_count={recovery_diagnostics['active_count']}",
        fontsize=13,
    )
    signal_fig = build_signal_figure(S, X, S_hat, fs, sample_count=args.signal_samples)
    dbscan_fig = build_dbscan_figure(A, A_hat, diagnostics)
    return fig, signal_fig, dbscan_fig


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize DBSCAN-Hough reproduction data and results.")
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
    parser.add_argument("--signal-samples", type=int, default=parameter.SIGNAL_SAMPLES)
    parser.add_argument("--source-preview-samples", type=int, default=parameter.SOURCE_PREVIEW_SAMPLES)
    parser.add_argument(
        "--tf-scatter-threshold",
        type=float,
        default=parameter.TF_SCATTER_THRESHOLD,
        help="Drop low-energy STFT bins from the raw STFT scatter plot.",
    )
    parser.add_argument("--save", default=parameter.DEFAULT_FIGURE_PATH)
    parser.add_argument("--no-show", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    S, A, X, fs = load_dataset(args.dataset, args.data_dir)
    figure, signal_figure, dbscan_figure = build_figure(S, A, X, fs, args)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")
        signal_save_path = save_path.with_name(f"{save_path.stem}_signals{save_path.suffix}")
        signal_figure.savefig(signal_save_path, dpi=200)
        print(f"Saved signal figure to {signal_save_path}")
        dbscan_save_path = save_path.with_name(f"{save_path.stem}_dbscan{save_path.suffix}")
        dbscan_figure.savefig(dbscan_save_path, dpi=200)
        print(f"Saved DBSCAN figure to {dbscan_save_path}")

    if not args.no_show:
        plt.show()

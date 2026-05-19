import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dbscan_hough
from main import load_dataset


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
        eps=args.eps,
        beta=args.beta,
        rho_tolerance=args.rho_tolerance,
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    time = np.arange(S.shape[1]) / fs
    for source_index, source in enumerate(S):
        axes[0, 0].plot(time[:2000], source[:2000], linewidth=0.9, label=f"s{source_index + 1}")
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
    time_limit = 1.05 * max(np.max(np.abs(X[0])), np.max(np.abs(X[1])), 1e-12)
    axes[0, 1].set_xlim(-time_limit, time_limit)
    axes[0, 1].set_ylim(-time_limit, time_limit)

    axes[1, 0].scatter(all_tf_points[:, 0], all_tf_points[:, 1], s=2, alpha=0.08, edgecolors="none")
    _draw_directions(axes[1, 0], A, color="tab:red", label_prefix="true")
    axes[1, 0].set_title("All STFT magnitude directions")
    axes[1, 0].set_xlabel("|X1(t, f)| normalized")
    axes[1, 0].set_ylabel("|X2(t, f)| normalized")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].grid(True, alpha=0.25)

    cluster_labels = sorted(label for label in np.unique(labels) if label != dbscan_hough.NOISE_LABEL)
    _draw_directions(axes[1, 1], A_hat, color="black", label_prefix="estimated")
    for label in cluster_labels:
        cluster_points = selected_points[labels == label]
        axes[1, 1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=8,
            alpha=0.75,
            edgecolors="none",
            label=f"cluster {label}",
        )
    if np.any(labels == dbscan_hough.NOISE_LABEL):
        noise = selected_points[labels == dbscan_hough.NOISE_LABEL]
        axes[1, 1].scatter(noise[:, 0], noise[:, 1], s=5, alpha=0.25, color="0.6", label="noise")
    axes[1, 1].set_title("Single-source TF points after DBSCAN-Hough")
    axes[1, 1].set_xlabel("|X1(t, f)| normalized")
    axes[1, 1].set_ylabel("|X2(t, f)| normalized")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].grid(True, alpha=0.25)

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"DBSCAN-Hough reproduction: {diagnostics['cluster_count']} sources estimated, "
        f"{diagnostics['single_source_count']} single-source TF points",
        fontsize=13,
    )
    return fig


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize DBSCAN-Hough reproduction data and results.")
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
    parser.add_argument("--save", default="figures/dbscan_hough_synthetic.png")
    parser.add_argument("--no-show", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    S, A, X, fs = load_dataset(args.dataset, args.data_dir)
    figure = build_figure(S, A, X, fs, args)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    if not args.no_show:
        plt.show()

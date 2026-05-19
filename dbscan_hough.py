from collections import deque
import math

import numpy as np
from scipy.signal import stft
from scipy.spatial import cKDTree


NOISE_LABEL = -1
UNVISITED_LABEL = -99


def orient_upper_half(points, eps=1e-12):
    points = np.asarray(points, dtype=float).copy()
    if points.ndim == 1:
        points = points.reshape(1, -1)

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, eps)

    pivots = np.argmax(np.abs(points), axis=1)
    signs = np.sign(points[np.arange(len(points)), pivots])
    signs[signs == 0] = 1
    return points * signs[:, None]


def compute_stft_matrix(X, fs, nperseg=512, noverlap=384, window="hann"):
    spectra = []
    for channel in X:
        _, _, Zxx = stft(
            channel,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )
        spectra.append(Zxx)
    return np.asarray(spectra)


def _phase_ratio_mask(Z, epsilon=0.06, real_floor=1e-8):
    real = Z.real
    imag = Z.imag
    valid = np.all(np.abs(real) > real_floor, axis=0)
    ratios = imag / np.where(np.abs(real) > real_floor, real, np.nan)

    mask = valid.copy()
    channel_count = Z.shape[0]
    for first in range(channel_count):
        for second in range(first + 1, channel_count):
            mask &= np.abs(ratios[first] - ratios[second]) < epsilon
    return mask


def _phase_angle_mask(Z, epsilon=0.06):
    phases = np.angle(Z)
    mask = np.ones(Z.shape[1:], dtype=bool)
    channel_count = Z.shape[0]
    for first in range(channel_count):
        for second in range(first + 1, channel_count):
            difference = np.angle(np.exp(1j * (phases[first] - phases[second])))
            mask &= np.abs(difference) < epsilon
    return mask


def detect_single_source_points(
    Z,
    epsilon=0.06,
    energy_fraction=0.1,
    phase_rule="angle",
):
    if phase_rule == "ratio":
        mask = _phase_ratio_mask(Z, epsilon=epsilon)
    elif phase_rule == "angle":
        mask = _phase_angle_mask(Z, epsilon=epsilon)
    else:
        raise ValueError("phase_rule must be 'angle' or 'ratio'")

    tf_vectors = Z.reshape(Z.shape[0], -1)
    mask = mask.reshape(-1)
    energies = np.linalg.norm(tf_vectors, axis=0)

    if np.any(mask):
        threshold = energy_fraction * np.max(energies[mask])
        mask &= energies > threshold

    return tf_vectors[:, mask], mask.reshape(Z.shape[1:])


def complex_tf_vectors_to_real_points(tf_vectors, eps=1e-12):
    if tf_vectors.size == 0:
        return np.empty((0, 0))

    magnitudes = np.abs(tf_vectors).T
    magnitudes = magnitudes / np.maximum(np.linalg.norm(magnitudes, axis=1, keepdims=True), eps)
    return orient_upper_half(magnitudes, eps=eps)


def estimate_eps(points, min_samples):
    points = np.asarray(points, dtype=float)
    point_count, dimension = points.shape
    if point_count == 0:
        raise ValueError("Cannot estimate Eps from an empty point set.")

    side_lengths = np.ptp(points, axis=0)
    side_lengths = np.maximum(side_lengths, 1e-12)
    rectangle_volume = float(np.prod(side_lengths))

    if dimension == 2:
        return np.sqrt(min_samples * rectangle_volume / (np.pi * point_count))
    if dimension == 3:
        return (3 * min_samples * rectangle_volume / (4 * np.pi * point_count)) ** (1 / 3)

    unit_ball_volume = np.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
    return (min_samples * rectangle_volume / (unit_ball_volume * point_count)) ** (1 / dimension)


def dbscan(points, eps, min_samples):
    points = np.asarray(points, dtype=float)
    labels = np.full(len(points), UNVISITED_LABEL, dtype=int)
    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, eps)
    cluster_id = 0

    for point_index in range(len(points)):
        if labels[point_index] != UNVISITED_LABEL:
            continue

        neighbors = neighborhoods[point_index]
        if len(neighbors) < min_samples:
            labels[point_index] = NOISE_LABEL
            continue

        labels[point_index] = cluster_id
        queue = deque(neighbors)
        while queue:
            neighbor_index = queue.popleft()
            if labels[neighbor_index] == NOISE_LABEL:
                labels[neighbor_index] = cluster_id
            if labels[neighbor_index] != UNVISITED_LABEL:
                continue

            labels[neighbor_index] = cluster_id
            neighbor_neighbors = neighborhoods[neighbor_index]
            if len(neighbor_neighbors) >= min_samples:
                queue.extend(neighbor_neighbors)

        cluster_id += 1

    return labels


def cluster_centers(points, labels):
    centers = []
    for label in sorted(label for label in np.unique(labels) if label != NOISE_LABEL):
        centers.append(points[labels == label].mean(axis=0))
    return orient_upper_half(np.asarray(centers))


def hough_refine_2d(points, beta=180, rho_tolerance=0.015):
    points = orient_upper_half(points)
    theta_values = np.linspace(0, np.pi, beta, endpoint=False)
    normal_vectors = np.column_stack([np.sin(theta_values), np.cos(theta_values)])
    rho = points @ normal_vectors.T
    votes = np.sum(np.abs(rho) <= rho_tolerance, axis=0)

    best_index = int(np.argmax(votes))
    theta = theta_values[best_index]
    direction = np.array([np.cos(theta), -np.sin(theta)], dtype=float)

    if np.dot(direction, points.mean(axis=0)) < 0:
        direction *= -1

    return orient_upper_half(direction)[0], votes, theta_values


def estimate_mixing_matrix(
    X,
    fs,
    nperseg=512,
    noverlap=384,
    epsilon=0.06,
    energy_fraction=0.1,
    min_samples=10,
    eps=None,
    beta=180,
    rho_tolerance=0.015,
    phase_rule="angle",
):
    Z = compute_stft_matrix(X, fs=fs, nperseg=nperseg, noverlap=noverlap)
    tf_vectors, single_source_mask = detect_single_source_points(
        Z,
        epsilon=epsilon,
        energy_fraction=energy_fraction,
        phase_rule=phase_rule,
    )
    points = complex_tf_vectors_to_real_points(tf_vectors)

    if len(points) == 0:
        raise RuntimeError("No single-source TF points were detected.")
    if points.shape[1] != 2:
        raise NotImplementedError("This reproduction implements the paper's 2-D Hough case.")

    if eps is None:
        eps = estimate_eps(points, min_samples=min_samples)

    labels = dbscan(points, eps=eps, min_samples=min_samples)
    cluster_labels = [label for label in sorted(np.unique(labels)) if label != NOISE_LABEL]
    if not cluster_labels:
        raise RuntimeError("DBSCAN did not find any clusters; try increasing eps or lowering min_samples.")

    columns = []
    hough_details = {}
    for label in cluster_labels:
        cluster_points = points[labels == label]
        direction, votes, theta_values = hough_refine_2d(
            cluster_points,
            beta=beta,
            rho_tolerance=rho_tolerance,
        )
        columns.append(direction)
        hough_details[int(label)] = {
            "point_count": int(len(cluster_points)),
            "max_votes": int(np.max(votes)),
            "theta": float(theta_values[int(np.argmax(votes))]),
        }

    A_hat = np.column_stack(columns)
    diagnostics = {
        "stft_shape": Z.shape,
        "single_source_count": int(tf_vectors.shape[1]),
        "point_count": int(len(points)),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "labels": labels,
        "cluster_count": int(len(cluster_labels)),
        "cluster_labels": [int(label) for label in cluster_labels],
        "cluster_sizes": {
            int(label): int(np.count_nonzero(labels == label)) for label in cluster_labels
        },
        "noise_count": int(np.count_nonzero(labels == NOISE_LABEL)),
        "hough": hough_details,
        "single_source_mask": single_source_mask,
        "points": points,
        "dbscan_centers": cluster_centers(points, labels),
    }
    return A_hat, diagnostics


def match_columns_by_angle(A, A_hat):
    A = orient_upper_half(A.T).T
    A_hat = orient_upper_half(A_hat.T).T
    source_count = A.shape[1]
    estimated_count = A_hat.shape[1]

    used = set()
    pairs = []
    angles = []
    for source_index in range(source_count):
        best = None
        for estimated_index in range(estimated_count):
            if estimated_index in used:
                continue
            cosine = np.dot(A[:, source_index], A_hat[:, estimated_index])
            cosine /= np.linalg.norm(A[:, source_index]) * np.linalg.norm(A_hat[:, estimated_index])
            cosine = np.clip(abs(cosine), -1.0, 1.0)
            angle = np.rad2deg(np.arccos(cosine))
            if best is None or angle < best[0]:
                best = (angle, estimated_index)

        if best is not None:
            angle, estimated_index = best
            used.add(estimated_index)
            pairs.append((source_index, estimated_index))
            angles.append(angle)

    return pairs, np.asarray(angles)

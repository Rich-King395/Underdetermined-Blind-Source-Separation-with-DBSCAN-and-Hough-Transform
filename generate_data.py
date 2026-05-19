import argparse
from pathlib import Path

import numpy as np


def _smooth_envelope(t, rate, phase):
    envelope = 0.65 + 0.25 * np.sin(2 * np.pi * rate * t + phase)
    envelope += 0.10 * np.sin(2 * np.pi * 0.37 * rate * t + 1.7 * phase)
    return np.clip(envelope, 0.05, None)


def _band_signal(t, freqs, rng, envelope_rate):
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    weights = rng.uniform(0.5, 1.0, size=len(freqs))
    signal = np.zeros_like(t)

    for weight, freq, phase in zip(weights, freqs, phases):
        vibrato = 2.5 * np.sin(2 * np.pi * 2.0 * t + phase / 3.0)
        instantaneous_phase = 2 * np.pi * freq * t
        instantaneous_phase += -(2.5 / 2.0) * np.cos(2 * np.pi * 2.0 * t + phase / 3.0)
        signal += weight * np.sin(instantaneous_phase + phase)

    signal *= _smooth_envelope(t, envelope_rate, phases[0])
    signal /= np.max(np.abs(signal)) + 1e-12
    return signal


def build_dataset(fs=8000, duration=5.0, seed=2026):
    rng = np.random.default_rng(seed)
    sample_count = int(fs * duration)
    t = np.arange(sample_count) / fs

    sources = np.vstack(
        [
            _band_signal(t, [260, 330, 410], rng, envelope_rate=0.9),
            _band_signal(t, [760, 930, 1110], rng, envelope_rate=0.7),
            _band_signal(t, [1580, 1860, 2180], rng, envelope_rate=0.5),
        ]
    )

    sources -= sources.mean(axis=1, keepdims=True)
    sources /= np.std(sources, axis=1, keepdims=True) + 1e-12

    angles = np.deg2rad([25, 55, 80])
    mixing_matrix = np.vstack([np.cos(angles), np.sin(angles)])
    observations = mixing_matrix @ sources
    observations /= np.max(np.abs(observations)) + 1e-12

    return sources, mixing_matrix, observations, fs


def save_dataset(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources, mixing_matrix, observations, fs = build_dataset()
    np.save(output_dir / "S_synthetic.npy", sources)
    np.save(output_dir / "A_synthetic.npy", mixing_matrix)
    np.save(output_dir / "X_synthetic.npy", observations)
    np.save(output_dir / "fs_synthetic.npy", np.array(fs))

    print("Saved synthetic DBSCAN-Hough dataset")
    print("Source matrix:", sources.shape)
    print("Mixing matrix:", mixing_matrix.shape)
    print("Observation matrix:", observations.shape)
    print("Sampling rate:", fs)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic UBSS dataset with time-overlapped but TF-sparse sources."
    )
    parser.add_argument("--output-dir", default="data")
    return parser


if __name__ == "__main__":
    save_dataset(build_parser().parse_args().output_dir)

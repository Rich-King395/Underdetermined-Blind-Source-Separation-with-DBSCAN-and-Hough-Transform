import argparse
from pathlib import Path

import numpy as np

import parameter


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


def _colored_noise(shape, rng, color=parameter.COLORED_NOISE_AR_COEFFICIENT):
    white = rng.standard_normal(shape)
    noise = np.zeros_like(white)
    noise[:, 0] = white[:, 0]
    for index in range(1, white.shape[1]):
        noise[:, index] = color * noise[:, index - 1] + (1 - color) * white[:, index]
    noise -= noise.mean(axis=1, keepdims=True)
    noise /= np.std(noise, axis=1, keepdims=True) + 1e-12
    return noise


def add_environment_noise(observations, rng, snr_db=parameter.OBSERVATION_SNR_DB):
    noise = parameter.WHITE_NOISE_WEIGHT * rng.standard_normal(observations.shape)
    noise += parameter.COLORED_NOISE_WEIGHT * _colored_noise(observations.shape, rng)
    signal_power = np.mean(observations**2)
    noise_power = np.mean(noise**2)
    scale = np.sqrt(signal_power / ((10 ** (snr_db / 10)) * noise_power + 1e-12))
    return observations + scale * noise


def build_dataset(
    fs=parameter.SAMPLE_RATE,
    duration=parameter.DURATION_SECONDS,
    seed=parameter.RANDOM_SEED,
    snr_db=parameter.OBSERVATION_SNR_DB,
):
    rng = np.random.default_rng(seed)
    sample_count = int(fs * duration)
    t = np.arange(sample_count) / fs

    sources = np.vstack(
        [
            _band_signal(t, freqs, rng, envelope_rate=rate)
            for freqs, rate in zip(parameter.SOURCE_FREQUENCY_BANDS, parameter.SOURCE_ENVELOPE_RATES)
        ]
    )

    shared = np.vstack(
        [
            weight * _band_signal(t, freqs, rng, envelope_rate=rate)
            for weight, freqs, rate in zip(
                parameter.SHARED_WEIGHTS,
                parameter.SHARED_FREQUENCY_BANDS,
                parameter.SHARED_ENVELOPE_RATES,
            )
        ]
    )
    sources += shared

    sources -= sources.mean(axis=1, keepdims=True)
    sources /= np.std(sources, axis=1, keepdims=True) + 1e-12

    angles = np.deg2rad(parameter.MIXING_ANGLES_DEGREES)
    mixing_matrix = np.vstack([np.cos(angles), np.sin(angles)])
    observations = mixing_matrix @ sources
    clean_observations = observations.copy()
    observations = add_environment_noise(observations, rng, snr_db=snr_db)
    observations /= np.max(np.abs(observations)) + 1e-12
    clean_observations /= np.max(np.abs(clean_observations)) + 1e-12

    return sources, mixing_matrix, observations, clean_observations, fs, snr_db


def save_dataset(output_dir, snr_db=parameter.OBSERVATION_SNR_DB):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources, mixing_matrix, observations, clean_observations, fs, snr_db = build_dataset(snr_db=snr_db)
    np.save(output_dir / "S_synthetic.npy", sources)
    np.save(output_dir / "A_synthetic.npy", mixing_matrix)
    np.save(output_dir / "X_synthetic.npy", observations)
    np.save(output_dir / "X_clean_synthetic.npy", clean_observations)
    np.save(output_dir / "fs_synthetic.npy", np.array(fs))
    np.save(output_dir / "snr_synthetic.npy", np.array(snr_db))

    print("Saved synthetic DBSCAN-Hough dataset")
    print("Source matrix:", sources.shape)
    print("Mixing matrix:", mixing_matrix.shape)
    print("Observation matrix:", observations.shape)
    print("Sampling rate:", fs)
    print("Observation SNR(dB):", snr_db)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic UBSS dataset with TF-overlapped sources and observation noise."
    )
    parser.add_argument("--output-dir", default=parameter.DATA_DIR)
    parser.add_argument("--snr-db", type=float, default=parameter.OBSERVATION_SNR_DB)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    save_dataset(args.output_dir, snr_db=args.snr_db)

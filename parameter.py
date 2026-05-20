"""Central configuration for the DBSCAN-Hough UBSS reproduction.

Edit this file to change the default dataset, algorithm, recovery, and plotting
parameters used by the project scripts.
"""

# Dataset paths and names
DATASET_NAME = "synthetic"
DATA_DIR = "data"
FIGURE_DIR = "figures"
FIGURE_BASENAME = "dbscan_hough_synthetic.png"

# Synthetic data generation
SAMPLE_RATE = 8000
DURATION_SECONDS = 5.0
RANDOM_SEED = 2026
OBSERVATION_SNR_DB = 30.0
SOURCE_FREQUENCY_BANDS = (
    (280, 430, 610, 820),
    (520, 700, 920, 1180),
    (760, 1040, 1320, 1680),
)
SOURCE_ENVELOPE_RATES = (0.9, 0.7, 0.5)
SHARED_FREQUENCY_BANDS = (
    (680, 980),
    (430, 1320),
    (610, 1180),
)
SHARED_WEIGHTS = (0.30, 0.25, 0.28)
SHARED_ENVELOPE_RATES = (0.35, 0.45, 0.55)
MIXING_ANGLES_DEGREES = (25, 55, 80)
WHITE_NOISE_WEIGHT = 0.65
COLORED_NOISE_WEIGHT = 0.35
COLORED_NOISE_AR_COEFFICIENT = 0.92

# STFT
STFT_NPERSEG = 512
STFT_NOVERLAP = 384
STFT_WINDOW = "hann"
STFT_BOUNDARY = "zeros"
STFT_PADDED = True

# Phase-angle single-source TF point detection
PHASE_EPSILON = 0.06
ENERGY_FRACTION = 0.1
PHASE_RULE = "angle"  # "angle" or "ratio"

# DBSCAN and Hough
DBSCAN_MIN_SAMPLES = 10
DBSCAN_EPS = None
MIN_CLUSTER_SIZE = None
MIN_CLUSTER_FRACTION = 0.15
HOUGH_BETA = 180
HOUGH_RHO_TOLERANCE = 0.015

# Source recovery
ACTIVE_COUNT = None

# Visualization
SIGNAL_SAMPLES = 4000
SOURCE_PREVIEW_SAMPLES = 2000
TF_SCATTER_THRESHOLD = 0.0
DEFAULT_FIGURE_PATH = f"{FIGURE_DIR}/{FIGURE_BASENAME}"

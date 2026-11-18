"""
Global configuration and constants for PathSnap (Team SI No 107).
"""
import os
import random
import numpy as np

# Project Root Directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure Matplotlib uses a writable config directory inside the project
MPL_DIR = os.path.join(PROJECT_ROOT, "outputs", ".mplconfig")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPL_DIR

# Team-specific reproducible random seed (Team SI No 107)
RANDOM_SEED = 107

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds across Python and NumPy for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Dataset directories
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data", "Student folder")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
STUDENT_TEST_DIR = os.path.join(DATA_ROOT, "student_test")

# Auto-resolve evaluator_test directory (supports root `evaluator_test/evaluator_test/`,
# root `evaluator_test/`, or `Data/Student folder/evaluator_test/`)
_EVAL_CANDIDATE_DIRS = [
    os.path.join(PROJECT_ROOT, "evaluator_test", "evaluator_test"),
    os.path.join(PROJECT_ROOT, "evaluator_test"),
    os.path.join(DATA_ROOT, "evaluator_test"),
]
EVALUATOR_TEST_DIR = next(
    (d for d in _EVAL_CANDIDATE_DIRS if os.path.exists(d) and any(os.path.isdir(os.path.join(d, sub)) and sub.isdigit() for sub in os.listdir(d))),
    _EVAL_CANDIDATE_DIRS[0],
)

# Output and Artifact directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EDA_OUT_DIR = os.path.join(OUTPUTS_DIR, "eda")
FIGURES_OUT_DIR = os.path.join(OUTPUTS_DIR, "figures")
MAPS_OUT_DIR = os.path.join(OUTPUTS_DIR, "maps")
PREDICTIONS_OUT_DIR = os.path.join(OUTPUTS_DIR, "predictions")
RESULTS_OUT_DIR = os.path.join(OUTPUTS_DIR, "results")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")

for _d in [
    MODELS_DIR,
    OUTPUTS_DIR,
    EDA_OUT_DIR,
    FIGURES_OUT_DIR,
    MAPS_OUT_DIR,
    PREDICTIONS_OUT_DIR,
    RESULTS_OUT_DIR,
    REPORT_DIR,
]:
    os.makedirs(_d, exist_ok=True)

# Earth radius in meters (WGS84 mean spherical radius)
EARTH_RADIUS_M = 6371008.8

# Candidate Generation Parameters
TOP_K_CANDIDATES = 8
MAX_CANDIDATE_RADIUS_M = 120.0  # meters

# Cross-Validation Settings
N_CV_FOLDS = 5

# Reserved 10 Evaluator Trip IDs (the 10 trips in 00000000..00000099 not in train or student_test)
# Plus support for any custom folder placed in evaluator_test or student_test
TEAM_INFO = {
    "team_id": "SI No 107",
    "members": [
        ("PES1UG24CS395", "Saatvik Gupta"),
        ("PES1UG24CS396", "Sagar A Walikar"),
        ("PES1UG24CS406", "Samarth Kotian"),
        ("PES1UG24CS407", "Samarth M H"),
    ],
}

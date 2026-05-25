from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Financial decision assumptions.
# These are simplified and should be clearly stated in the paper.
LOSS_GIVEN_DEFAULT = 0.60
PROFIT_RATE_IF_REPAID = 0.08

# Classification threshold.
DEFAULT_THRESHOLD = 0.50

# Stress-test assumptions.
STRESS_BILL_INCREASE = 1.15
STRESS_PAYMENT_DECREASE = 0.75
STRESS_LIMIT_DECREASE = 0.90
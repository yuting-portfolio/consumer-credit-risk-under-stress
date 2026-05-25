import numpy as np
import pandas as pd
from src.config import RESULTS_DIR, DEFAULT_THRESHOLD, LOSS_GIVEN_DEFAULT

def bootstrap_loss_ci(part, n_boot=500, random_state=42):
    rng = np.random.default_rng(random_state)
    losses = []

    y = part["y_true"].values.astype(int)
    p = part["y_prob_default"].values
    exposure = part["limit_bal"].clip(lower=0).values if "limit_bal" in part.columns else np.ones(len(part)) * 10000
    n = len(part)

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yb = y[idx]
        pb = p[idx]
        eb = exposure[idx]

        approve = pb < DEFAULT_THRESHOLD
        false_approval = approve & (yb == 1)
        loss = np.sum(eb[false_approval] * LOSS_GIVEN_DEFAULT)
        losses.append(loss / n)

    return {
        "mean_avg_loss": float(np.mean(losses)),
        "lower_95": float(np.percentile(losses, 2.5)),
        "upper_95": float(np.percentile(losses, 97.5))
    }

def run_bootstrap():
    path = RESULTS_DIR / "predictions_with_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run src/models.py first.")

    df = pd.read_csv(path)
    rows = []

    for model_name, part in df.groupby("model"):
        ci = bootstrap_loss_ci(part)
        ci["model"] = model_name
        rows.append(ci)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "bootstrap_loss_ci.csv", index=False)
    print("Saved bootstrap loss confidence intervals.")
    return out

if __name__ == "__main__":
    run_bootstrap()
import numpy as np
import pandas as pd
from src.config import RESULTS_DIR, LOSS_GIVEN_DEFAULT, PROFIT_RATE_IF_REPAID


def _expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        avg_prob = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        gap = abs(avg_prob - avg_true)
        weight = mask.mean()
        ece += weight * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def _financial_value(y_true, y_prob, exposure, threshold):
    approve = y_prob < threshold
    gain_repaid = approve & (y_true == 0)
    loss_default = approve & (y_true == 1)
    profit = np.sum(exposure[gain_repaid] * PROFIT_RATE_IF_REPAID)
    loss = np.sum(exposure[loss_default] * LOSS_GIVEN_DEFAULT)
    return float(profit - loss), float(loss / len(y_true))


def _threshold_search(y_true, y_prob, exposure):
    rows = []
    for thr in np.linspace(0.05, 0.95, 91):
        approve = y_prob < thr
        default_mask = y_true == 1
        safe_mask = y_true == 0
        false_approval_rate = (approve & default_mask).sum() / max(default_mask.sum(), 1)
        false_rejection_rate = ((~approve) & safe_mask).sum() / max(safe_mask.sum(), 1)
        net_value, avg_loss = _financial_value(y_true, y_prob, exposure, thr)
        rows.append({
            "threshold": float(thr),
            "approval_rate": float(approve.mean()),
            "false_approval_rate": float(false_approval_rate),
            "false_rejection_rate": float(false_rejection_rate),
            "avg_loss_per_applicant": avg_loss,
            "avg_net_value_per_applicant": float(net_value / len(y_true))
        })
    return pd.DataFrame(rows)


def _paired_bootstrap_diff(base, challenger, n_boot=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(base)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        diffs.append(challenger[idx].mean() - base[idx].mean())
    diffs = np.asarray(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_advanced_analysis():
    path = RESULTS_DIR / "predictions_with_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run src/models.py first.")

    df = pd.read_csv(path)

    calibration_rows = []
    threshold_rows = []
    model_value_vectors = {}

    for model_name, part in df.groupby("model"):
        y_true = part["y_true"].astype(int).values
        y_prob = part["y_prob_default"].astype(float).values
        if "limit_bal" in part.columns:
            exposure = part["limit_bal"].clip(lower=0).astype(float).values
        else:
            exposure = np.ones(len(part), dtype=float) * 10000.0

        ece, mce = _expected_calibration_error(y_true, y_prob, n_bins=10)
        brier = float(np.mean((y_true - y_prob) ** 2))
        calibration_rows.append({
            "model": model_name,
            "brier_score": brier,
            "expected_calibration_error": ece,
            "max_calibration_error": mce
        })

        sweep = _threshold_search(y_true, y_prob, exposure)
        sweep["model"] = model_name
        threshold_rows.append(sweep)

        # Per-applicant realized value vector for paired model comparison.
        value_vector = np.where(
            y_prob < 0.5,
            np.where(y_true == 0, exposure * PROFIT_RATE_IF_REPAID, -exposure * LOSS_GIVEN_DEFAULT),
            0.0
        )
        model_value_vectors[model_name] = value_vector

    calibration_df = pd.DataFrame(calibration_rows).sort_values("brier_score")
    calibration_df.to_csv(RESULTS_DIR / "model_calibration.csv", index=False)

    threshold_df = pd.concat(threshold_rows, ignore_index=True)
    threshold_df.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)

    best_thresholds = []
    for model_name, part in threshold_df.groupby("model"):
        # Choose high-value threshold while capping false approvals.
        constrained = part[part["false_approval_rate"] <= 0.35]
        selected = constrained if len(constrained) > 0 else part
        best = selected.sort_values("avg_net_value_per_applicant", ascending=False).iloc[0]
        best_thresholds.append(best.to_dict())
    best_thresholds_df = pd.DataFrame(best_thresholds).sort_values("avg_net_value_per_applicant", ascending=False)
    best_thresholds_df.to_csv(RESULTS_DIR / "threshold_optimization.csv", index=False)

    comparison_rows = []
    model_names = sorted(model_value_vectors.keys())
    for i, base_name in enumerate(model_names):
        for challenger_name in model_names[i + 1:]:
            base = model_value_vectors[base_name]
            challenger = model_value_vectors[challenger_name]
            mean_diff, lo, hi = _paired_bootstrap_diff(base, challenger, n_boot=1000, random_state=42)
            comparison_rows.append({
                "base_model": base_name,
                "challenger_model": challenger_name,
                "mean_diff_avg_value_per_applicant": mean_diff,
                "diff_ci_lower_95": lo,
                "diff_ci_upper_95": hi,
                "is_significant_at_95": bool(lo > 0 or hi < 0)
            })
    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "model_comparison_bootstrap.csv", index=False)
    print("Saved advanced analysis outputs.")


if __name__ == "__main__":
    run_advanced_analysis()

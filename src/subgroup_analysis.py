import pandas as pd
import numpy as np
from src.config import RESULTS_DIR, DEFAULT_THRESHOLD, LOSS_GIVEN_DEFAULT, PROFIT_RATE_IF_REPAID

def create_subgroups(df):
    df = df.copy()

    if "limit_bal" in df.columns:
        df["group_low_credit_limit"] = df["limit_bal"] <= df["limit_bal"].median()
    else:
        df["group_low_credit_limit"] = False

    if "age" in df.columns:
        df["group_young_borrower"] = df["age"] <= 30
        df["group_older_borrower"] = df["age"] >= 55
    else:
        df["group_young_borrower"] = False
        df["group_older_borrower"] = False

    if "num_delayed_months" in df.columns:
        df["group_prior_payment_delay"] = df["num_delayed_months"] >= 2
    else:
        repayment_cols = [c for c in df.columns if c.startswith("pay_") and not c.startswith("pay_amt")]
        if repayment_cols:
            df["group_prior_payment_delay"] = (df[repayment_cols] > 0).sum(axis=1) >= 2
        else:
            df["group_prior_payment_delay"] = False

    if "bill_to_limit_ratio" in df.columns:
        df["group_high_utilization"] = df["bill_to_limit_ratio"] >= df["bill_to_limit_ratio"].median()
    else:
        df["group_high_utilization"] = False

    return df

def subgroup_value_metrics(df, group_col):
    group = df[df[group_col] == True]
    reference = df[df[group_col] == False]

    if len(group) < 50 or len(reference) < 50:
        return None

    def calc(part):
        exposure = part["limit_bal"].clip(lower=0).values if "limit_bal" in part.columns else np.ones(len(part)) * 10000
        approve = part["y_prob_default"].values < DEFAULT_THRESHOLD
        y = part["y_true"].values.astype(int)

        false_approval = approve & (y == 1)
        false_rejection = (~approve) & (y == 0)

        loss = np.sum(exposure[false_approval] * LOSS_GIVEN_DEFAULT)
        missed_profit = np.sum(exposure[false_rejection] * PROFIT_RATE_IF_REPAID)

        return {
            "n": len(part),
            "approval_rate": approve.mean(),
            "false_approval_rate": false_approval.mean(),
            "false_rejection_rate": false_rejection.mean(),
            "realized_default_loss": loss,
            "missed_profit_from_rejection": missed_profit,
            "avg_loss_per_applicant": loss / len(part)
        }

    g = calc(group)
    r = calc(reference)

    return {
        "subgroup": group_col,
        "n_subgroup": g["n"],
        "n_reference": r["n"],
        "subgroup_approval_rate": g["approval_rate"],
        "reference_approval_rate": r["approval_rate"],
        "approval_rate_gap": abs(g["approval_rate"] - r["approval_rate"]),
        "subgroup_false_approval_rate": g["false_approval_rate"],
        "reference_false_approval_rate": r["false_approval_rate"],
        "false_approval_gap": abs(g["false_approval_rate"] - r["false_approval_rate"]),
        "subgroup_false_rejection_rate": g["false_rejection_rate"],
        "reference_false_rejection_rate": r["false_rejection_rate"],
        "false_rejection_gap": abs(g["false_rejection_rate"] - r["false_rejection_rate"]),
        "subgroup_avg_loss_per_applicant": g["avg_loss_per_applicant"],
        "reference_avg_loss_per_applicant": r["avg_loss_per_applicant"],
        "loss_gap_per_applicant": abs(g["avg_loss_per_applicant"] - r["avg_loss_per_applicant"])
    }

def run_subgroup_analysis():
    path = RESULTS_DIR / "predictions_with_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run src/models.py first.")

    df = pd.read_csv(path)
    df = create_subgroups(df)

    group_cols = [
        "group_low_credit_limit",
        "group_young_borrower",
        "group_older_borrower",
        "group_prior_payment_delay",
        "group_high_utilization"
    ]

    rows = []
    for model_name, model_df in df.groupby("model"):
        for group_col in group_cols:
            row = subgroup_value_metrics(model_df, group_col)
            if row is not None:
                row["model"] = model_name
                rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["model", "loss_gap_per_applicant"], ascending=[True, False])
    out.to_csv(RESULTS_DIR / "subgroup_metrics.csv", index=False)
    print("Saved subgroup metrics.")
    return out

if __name__ == "__main__":
    run_subgroup_analysis()
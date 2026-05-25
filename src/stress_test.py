import pandas as pd
import numpy as np
import joblib
from src.config import RESULTS_DIR, STRESS_BILL_INCREASE, STRESS_PAYMENT_DECREASE, STRESS_LIMIT_DECREASE, DEFAULT_THRESHOLD
from src.data_loader import load_credit_card_default
from src.features import get_feature_target
from src.metrics import prediction_metrics, expected_financial_value, exposure_from_features

def apply_economic_stress(X):
    '''
    Simulates economic stress by worsening repayment capacity:
    - credit limits decline
    - bill amounts increase
    - payment amounts decrease
    - repayment-delay proxy features increase if present

    This is not a causal macroeconomic model. It is a controlled stress test.
    '''
    Xs = X.copy()

    for col in Xs.columns:
        if col == "limit_bal":
            Xs[col] = Xs[col] * STRESS_LIMIT_DECREASE
        elif col.startswith("bill_amt") or col in ["avg_bill_amt", "max_bill_amt"]:
            Xs[col] = Xs[col] * STRESS_BILL_INCREASE
        elif col.startswith("pay_amt") or col in ["avg_payment_amt", "total_payment_amt"]:
            Xs[col] = Xs[col] * STRESS_PAYMENT_DECREASE
        elif col in ["bill_to_limit_ratio"]:
            Xs[col] = Xs[col] * (STRESS_BILL_INCREASE / STRESS_LIMIT_DECREASE)
        elif col in ["payment_to_bill_ratio"]:
            Xs[col] = Xs[col] * (STRESS_PAYMENT_DECREASE / STRESS_BILL_INCREASE)

    return Xs

def run_stress_test():
    df = load_credit_card_default()
    X, y, enriched = get_feature_target(df)

    model_files = list(RESULTS_DIR.glob("*.joblib"))
    rows = []

    for model_path in model_files:
        model_name = model_path.stem
        model = joblib.load(model_path)

        X_stress = apply_economic_stress(X)
        y_prob = model.predict_proba(X_stress)[:, 1]

        pred_metrics = prediction_metrics(y.values, y_prob, threshold=DEFAULT_THRESHOLD)
        exposure = exposure_from_features(X_stress)
        value_metrics = expected_financial_value(y.values, y_prob, exposure, threshold=DEFAULT_THRESHOLD)

        row = {"model": model_name, "scenario": "economic_stress"}
        row.update(pred_metrics)
        row.update(value_metrics)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "stress_test_metrics.csv", index=False)
    print("Saved stress test metrics.")
    return out

if __name__ == "__main__":
    run_stress_test()
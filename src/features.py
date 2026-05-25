import numpy as np
import pandas as pd

def add_engineered_features(df):
    df = df.copy()

    # Expected UCI columns after standardization:
    # limit_bal, age, pay_0/pay_2..., bill_amt1..., pay_amt1...
    bill_cols = [c for c in df.columns if c.startswith("bill_amt")]
    pay_amt_cols = [c for c in df.columns if c.startswith("pay_amt")]
    repayment_cols = [c for c in df.columns if c.startswith("pay_") and not c.startswith("pay_amt")]

    if "limit_bal" in df.columns:
        df["log_limit_bal"] = np.log1p(df["limit_bal"].clip(lower=0))
    else:
        df["log_limit_bal"] = 0

    if bill_cols:
        df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
        df["max_bill_amt"] = df[bill_cols].max(axis=1)
        if "limit_bal" in df.columns:
            df["bill_to_limit_ratio"] = df["avg_bill_amt"] / (df["limit_bal"].replace(0, np.nan))
            df["bill_to_limit_ratio"] = df["bill_to_limit_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        df["avg_bill_amt"] = 0
        df["max_bill_amt"] = 0
        df["bill_to_limit_ratio"] = 0

    if pay_amt_cols:
        df["avg_payment_amt"] = df[pay_amt_cols].mean(axis=1)
        df["total_payment_amt"] = df[pay_amt_cols].sum(axis=1)
        df["payment_to_bill_ratio"] = df["avg_payment_amt"] / (df["avg_bill_amt"].replace(0, np.nan))
        df["payment_to_bill_ratio"] = df["payment_to_bill_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        df["avg_payment_amt"] = 0
        df["total_payment_amt"] = 0
        df["payment_to_bill_ratio"] = 0

    if repayment_cols:
        df["avg_repayment_delay"] = df[repayment_cols].mean(axis=1)
        df["max_repayment_delay"] = df[repayment_cols].max(axis=1)
        df["num_delayed_months"] = (df[repayment_cols] > 0).sum(axis=1)
    else:
        df["avg_repayment_delay"] = 0
        df["max_repayment_delay"] = 0
        df["num_delayed_months"] = 0

    return df

def get_feature_target(df):
    df = add_engineered_features(df)
    y = df["default"].astype(int)
    X = df.drop(columns=["default"])
    return X, y, df
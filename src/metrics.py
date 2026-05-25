import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.config import LOSS_GIVEN_DEFAULT, PROFIT_RATE_IF_REPAID, DEFAULT_THRESHOLD

def prediction_metrics(y_true, y_prob, threshold=DEFAULT_THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # In this project:
    # positive class = default risk.
    # Decision rule: approve if predicted non-default.
    # false approval = predicted safe but actually default = FN.
    # false rejection = predicted default but actually safe = FP.
    false_approval_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    false_rejection_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    out.update({
        "tn": tn,
        "fp_false_rejection": fp,
        "fn_false_approval": fn,
        "tp": tp,
        "false_approval_rate": false_approval_rate,
        "false_rejection_rate": false_rejection_rate
    })

    return out

def expected_financial_value(y_true, y_prob, credit_exposure, threshold=DEFAULT_THRESHOLD):
    '''
    Simplified lending decision model.

    If predicted default probability >= threshold, reject applicant.
    If predicted default probability < threshold, approve applicant.

    If approved and actually repaid: gain = exposure * profit rate.
    If approved and actually defaulted: loss = exposure * LGD.
    If rejected: financial value = 0 in this simplified model.

    Returns total expected realized value and total realized loss.
    '''
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    exposure = np.asarray(credit_exposure).astype(float)

    approve = y_prob < threshold

    gain_repaid = approve & (y_true == 0)
    loss_default = approve & (y_true == 1)

    profit = np.sum(exposure[gain_repaid] * PROFIT_RATE_IF_REPAID)
    loss = np.sum(exposure[loss_default] * LOSS_GIVEN_DEFAULT)
    net_value = profit - loss

    return {
        "approved_count": int(approve.sum()),
        "approval_rate": float(approve.mean()),
        "gross_profit_if_repaid": float(profit),
        "realized_default_loss": float(loss),
        "net_value": float(net_value),
        "avg_loss_per_applicant": float(loss / len(y_true)),
        "avg_net_value_per_applicant": float(net_value / len(y_true))
    }

def exposure_from_features(X):
    if "limit_bal" in X.columns:
        return X["limit_bal"].clip(lower=0).values
    if "loan" in X.columns:
        return X["loan"].clip(lower=0).values
    return np.ones(len(X)) * 10000
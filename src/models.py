import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import RANDOM_STATE, TEST_SIZE, RESULTS_DIR, DEFAULT_THRESHOLD
from src.data_loader import load_credit_card_default, dataset_summary
from src.features import get_feature_target
from src.metrics import prediction_metrics, expected_financial_value, exposure_from_features

def build_preprocessor(X):
    numeric_cols = list(X.columns)
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols)
    ])

def build_models(X):
    preprocessor = build_preprocessor(X)

    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),
        "gradient_boosting": Pipeline([
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ])
    }

def get_prob(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))

def train_and_evaluate():
    df = load_credit_card_default()
    summary = dataset_summary(df)
    pd.DataFrame([{
        "n_rows": summary["n_rows"],
        "n_columns": summary["n_columns"],
        "default_rate": summary["default_rate"]
    }]).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    X, y, enriched_df = get_feature_target(df)

    X_train, X_test, y_train, y_test, enriched_train, enriched_test = train_test_split(
        X, y, enriched_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models(X_train)
    rows = []
    prediction_frames = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_prob = get_prob(model, X_test)
        pred_metrics = prediction_metrics(y_test.values, y_prob, threshold=DEFAULT_THRESHOLD)
        exposure = exposure_from_features(X_test)
        value_metrics = expected_financial_value(y_test.values, y_prob, exposure, threshold=DEFAULT_THRESHOLD)

        row = {"model": name}
        row.update(pred_metrics)
        row.update(value_metrics)
        rows.append(row)

        pred_df = enriched_test.copy()
        pred_df["record_id"] = enriched_test.index
        pred_df["model"] = name
        pred_df["y_true"] = y_test.values
        pred_df["y_prob_default"] = y_prob
        pred_df["y_pred_default"] = (y_prob >= DEFAULT_THRESHOLD).astype(int)
        pred_df["approved"] = (y_prob < DEFAULT_THRESHOLD).astype(int)
        prediction_frames.append(pred_df)

        joblib.dump(model, RESULTS_DIR / f"{name}.joblib")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    preds = pd.concat(prediction_frames, ignore_index=True)
    preds.to_csv(RESULTS_DIR / "predictions_with_features.csv", index=False)

    print("Saved model metrics and predictions.")
    return metrics_df, preds

if __name__ == "__main__":
    train_and_evaluate()
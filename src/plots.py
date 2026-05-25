import pandas as pd
import matplotlib.pyplot as plt
from src.config import RESULTS_DIR, FIGURES_DIR

def plot_model_accuracy():
    df = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["accuracy"])
    plt.title("Model Accuracy on Credit Default Prediction")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_accuracy.png", dpi=200)
    plt.close()

def plot_expected_loss():
    df = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["avg_loss_per_applicant"])
    plt.title("Average Default Loss per Applicant")
    plt.ylabel("Average Loss")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "expected_loss.png", dpi=200)
    plt.close()

def plot_false_approval_rate():
    df = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["false_approval_rate"])
    plt.title("False Approval Rate by Model")
    plt.ylabel("False Approval Rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "false_approval_rate.png", dpi=200)
    plt.close()

def plot_stress_loss():
    base = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    stress_path = RESULTS_DIR / "stress_test_metrics.csv"
    if not stress_path.exists():
        return

    stress = pd.read_csv(stress_path)
    merged = base[["model", "avg_loss_per_applicant"]].merge(
        stress[["model", "avg_loss_per_applicant"]],
        on="model",
        suffixes=("_baseline", "_stress")
    )

    x = range(len(merged))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([i - width/2 for i in x], merged["avg_loss_per_applicant_baseline"], width=width, label="Baseline")
    plt.bar([i + width/2 for i in x], merged["avg_loss_per_applicant_stress"], width=width, label="Economic Stress")
    plt.title("Expected Loss: Baseline vs Economic Stress")
    plt.ylabel("Average Loss per Applicant")
    plt.xticks(list(x), merged["model"], rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stress_loss_comparison.png", dpi=200)
    plt.close()

def plot_subgroup_loss():
    path = RESULTS_DIR / "subgroup_metrics.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    top = df.sort_values("loss_gap_per_applicant", ascending=False).head(10)
    labels = top["model"] + " / " + top["subgroup"]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, top["loss_gap_per_applicant"])
    plt.title("Largest Subgroup Loss Gaps")
    plt.ylabel("Loss Gap per Applicant")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "subgroup_loss.png", dpi=200)
    plt.close()

def plot_bootstrap_ci():
    path = RESULTS_DIR / "bootstrap_loss_ci.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    x = range(len(df))
    yerr_lower = df["mean_avg_loss"] - df["lower_95"]
    yerr_upper = df["upper_95"] - df["mean_avg_loss"]

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, df["mean_avg_loss"], yerr=[yerr_lower, yerr_upper], fmt="o", capsize=5)
    plt.title("Bootstrap 95% CI for Average Loss")
    plt.ylabel("Average Loss per Applicant")
    plt.xticks(list(x), df["model"], rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bootstrap_loss_ci.png", dpi=200)
    plt.close()


def plot_threshold_tradeoff():
    path = RESULTS_DIR / "threshold_optimization.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("avg_net_value_per_applicant", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["threshold"])
    plt.title("Financially Optimal Threshold by Model")
    plt.ylabel("Selected Threshold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optimal_thresholds.png", dpi=200)
    plt.close()


def plot_calibration_error():
    path = RESULTS_DIR / "model_calibration.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values("expected_calibration_error", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["expected_calibration_error"])
    plt.title("Expected Calibration Error (ECE) by Model")
    plt.ylabel("ECE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_ece.png", dpi=200)
    plt.close()

def generate_plots():
    plot_model_accuracy()
    plot_expected_loss()
    plot_false_approval_rate()
    plot_stress_loss()
    plot_subgroup_loss()
    plot_bootstrap_ci()
    plot_threshold_tradeoff()
    plot_calibration_error()
    print("Saved figures.")

if __name__ == "__main__":
    generate_plots()
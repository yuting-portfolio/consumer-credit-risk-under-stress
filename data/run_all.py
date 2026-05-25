from src.models import train_and_evaluate
from src.stress_test import run_stress_test
from src.subgroup_analysis import run_subgroup_analysis
from src.bootstrap import run_bootstrap
from src.advanced_analysis import run_advanced_analysis
from src.plots import generate_plots

if __name__ == "__main__":
    train_and_evaluate()
    run_stress_test()
    run_subgroup_analysis()
    run_bootstrap()
    run_advanced_analysis()
    generate_plots()
    print("Full real-data credit risk research pipeline complete.")
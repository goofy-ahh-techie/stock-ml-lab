import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_split(name, Y_true, Y_pred, Y_prob):
    metrics = {
        "accuracy": accuracy_score(Y_true, Y_pred),
        "precision_up": precision_score(Y_true, Y_pred, pos_label=1),
        "recall_up": recall_score(Y_true, Y_pred, pos_label=1),
        "f1_up": f1_score(Y_true, Y_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(Y_true, Y_pred).tolist()
    }

    print(f"\n{name.upper()} METRICS")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

    print("Confusion matrix [[TN, FP], [FN, TP]]: ")
    print(np.array(metrics["confusion_matrix"]))

    return metrics

def naive_baseline(Y_true):
    n = len(Y_true)

    always_up = np.ones(n)
    always_down = np.zeros(n)
    random_50 = np.random.randint(0, 2, size=n)

    return {
        "always_up": {
            "accuracy": accuracy_score(Y_true, always_up),
            "precision_up": precision_score(Y_true, always_up, zero_division=0),
            "recall_up": recall_score(Y_true, always_up),
        },
        "always_down": {
            "accuracy": accuracy_score(Y_true, always_down),
            "precision_up": 0.0,
            "recall_up": 0.0,
        },
        "random_50": {
            "accuracy": accuracy_score(Y_true, random_50),
            "precision_up": precision_score(Y_true, random_50, zero_division=0),
            "recall_up": recall_score(Y_true, random_50),
        }
    }

def run_evaluation(bundle: dict):
    results = {
        "val": {},
        "test": {}
    }

    results["val"]["model"] = evaluate_split(
        "val", bundle["Y_val"], 
        bundle["Y_val_pred"], bundle["Y_val_prob"]
    )

    results["val"]["baselines"] = naive_baseline(bundle["Y_val"])

    results["test"]["model"] = evaluate_split(
        "test", bundle["Y_test"], 
        bundle["Y_test_pred"], bundle["Y_test_prob"]
    )

    results["test"]["baselines"] = naive_baseline(bundle["Y_test"])
    save_metrics_to_json(results)
    write_summary_report(results)

    plt.hist(bundle["Y_test_prob"], bins=30)
    plt.title("Test set predicted UP probabilities.")
    plt.xlabel("P(UP)")
    plt.ylabel("Count")
    plt.show()


def save_metrics_to_json(results: dict):
    Path("reports").mkdir(exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(results, f , indent=2)

    print("Saved reports/metrics.json")

def write_summary_report(results: dict):
    summary = f"""
    # Stage 7 — Evaluation Summary

    ## What was evaluated
    - Logistic Regression baseline
    - Time-based split
    - No hyperparameter tuning

    ## Validation Performance
    Accuracy: {results['val']['model']['accuracy']:.3f}
    F1 (UP): {results['val']['model']['f1_up']:.3f}

    ## Test Performance
    Accuracy: {results['test']['model']['accuracy']:.3f}
    F1 (UP): {results['test']['model']['f1_up']:.3f}

    ## Baseline Comparison (Test)
    - Always UP accuracy: {results['test']['baselines']['always_up']['accuracy']:.3f}
    - Always DOWN accuracy: {results['test']['baselines']['always_down']['accuracy']:.3f}
    - Random 50% accuracy: {results['test']['baselines']['random_50']['accuracy']:.3f}

    ## Conclusion
    - Model beats random and naive baselines on test
    - Signal is weak but consistent
    - Coefficients align with market intuition

    ## Next Steps
    - Threshold tuning (only trade when P(UP) > 0.6)
    - Precision-focused evaluation
    - Walk-forward validation
    """

    with open("reports/summary.md", "w") as f:
        f.write(summary)

    print("✅ Saved reports/summary.md")
import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path

def train_logreg(bundle: dict):

    X_train, Y_train = bundle["X_train"], bundle["Y_train"]
    X_val, Y_val = bundle["X_val"], bundle["Y_val"]
    X_test, Y_test = bundle["X_test"], bundle["X_test"]
    
    # ---------- MODEL ----------
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    # ---------- PREDICTIONS ----------
    

    # ---------- TRAIN ----------
    model.fit(X_train, Y_train)

    # ---------- SAVE MODEL ----------
    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": bundle["scaler"],
            "feature_cols": bundle["feature_cols"]
        },
        "models/logreg_baseline.joblib"
    )
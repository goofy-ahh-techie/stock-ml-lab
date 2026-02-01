import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def train_log_reg(bundle: dict):
    X_train, Y_train = bundle["X_train"], bundle["Y_train"]
    X_val, Y_val = bundle["X_val"], bundle["Y_val"]
    X_test, Y_test = bundle["X_test"], bundle["Y_test"]
    
    # ---------- MODEL ----------
    model = LogisticRegression(
        # penalty="l2",
        # solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    # ---------- TRAIN ----------
    model.fit(X_train, Y_train)

    # ---------- PREDICTIONS ----------
    Y_val_pred = model.predict(X_val)
    Y_val_prob = model.predict_proba(X_val)[:, 1]

    Y_test_pred = model.predict(X_test)
    Y_test_prob = model.predict_proba(X_test)[:, 1]

    bundle["Y_val_pred"], bundle["Y_val_prob"] = Y_val_pred, Y_val_prob
    bundle["Y_test_pred"], bundle["Y_test_prob"] = Y_test_pred, Y_test_prob

    # ---------- METRICS ----------
    print("\nValidation Performance")
    print("Accuracy: ", accuracy_score(Y_val, Y_val_pred))
    print(classification_report(Y_val, Y_val_pred, digits=4))

    print("\nTest Performance")
    print("Accuracy: ", accuracy_score(Y_test, Y_test_pred))
    print(classification_report(Y_test, Y_test_pred, digits=4))

    # ---------- FEATURE IMPORTANCE ----------
    coef_df = pd.DataFrame({
        "feature": bundle["feature_cols"],
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", ascending=False)

    print("\nTop Positive Coefficients (Bullish signals)")
    print(coef_df.head(3))

    print("\nTop Negative Coefficients (Bearish signals)")
    print(coef_df.tail(3))

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
    print("\nModel saved to models/logreg_baseline.joblib")

    return model, bundle
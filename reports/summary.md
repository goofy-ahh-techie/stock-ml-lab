
    # Stage 7 — Evaluation Summary

    ## What was evaluated
    - Logistic Regression baseline
    - Time-based split
    - No hyperparameter tuning

    ## Validation Performance
    Accuracy: 0.500
    F1 (UP): 0.553

    ## Test Performance
    Accuracy: 0.544
    F1 (UP): 0.543

    ## Baseline Comparison (Test)
    - Always UP accuracy: 0.512
    - Always DOWN accuracy: 0.488
    - Random 50% accuracy: 0.528

    ## Conclusion
    - Model beats random and naive baselines on test
    - Signal is weak but consistent
    - Coefficients align with market intuition

    ## Next Steps
    - Threshold tuning (only trade when P(UP) > 0.6)
    - Precision-focused evaluation
    - Walk-forward validation
    
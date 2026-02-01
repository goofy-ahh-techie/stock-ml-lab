Some basic commands to run on doing the basic setup for running the application, as we are using poetry for env. 

1. python --version  
2. pip --version
3. pip install poetry

4. poetry config virtualenvs.in-project true
5. poetry init 
6. poetry env use python
7. poetry add pandas numpy scikit-learn matplotlib yfinance pyarrow
8. poetry add --group dev jupyter ipykernal ruff black pytest

---- Now, go to puproject.toml, add this at bottom(we don't wanna use poetry for packages) : 
[tool.poetry]
package-mode = false

9. poetry install



📈 Stock Direction Prediction — Baseline ML Pipeline
1. Problem Statement

The goal of this project is to build a clean, leakage-free baseline machine learning pipeline to predict next-day stock price direction (UP/DOWN) using only past market data.

This project is intentionally focused on:

correctness over complexity

interpretability over black-box models

realistic evaluation over inflated metrics

The output is not a trading bot, but a probability signal that can later be converted into a trading strategy.

2. Data Source

Market: Indian equities

Frequency: Daily OHLCV candles

Source: Yahoo Finance (free, adjusted prices)

Fields used:

Open, High, Low, Close, Volume

Data is cleaned to:

remove missing OHLCV rows

ensure ascending time order

drop last row where next-day label is unavailable

3. Label Definition (No Leakage)

Target (y):
Binary next-day direction

y[t] = 1  if Close[t+1] > Close[t]
y[t] = 0  otherwise


Key point:

Labels use future data

Features use only current & past data

No overlap → no lookahead bias

4. Feature Engineering

Minimal, interpretable feature set (past-only):

Price Momentum

ret_1 → 1-day return

ret_5 → 5-day return

ret_10 → 10-day return

Volatility

vol_5 → rolling std of ret_1 (5 days)

Volume

vol_chg_1 → day-over-day volume change

vol_ratio_5 → volume / 5-day average volume

All rows affected by rolling windows are dropped.

Final dataset:

Date + 6 features + y

5. Train / Validation / Test Split (Time-Based)

No random shuffling is used.

Split	Period
Train	2015
Validation	2016
Test	2017

Rules enforced:

max(train_date) < min(val_date) < min(test_date)

Scaling is fit only on train, then applied to val/test

This simulates real future deployment.

6. Model
Baseline Model

Logistic Regression

L2 regularization

Interpretable coefficients

Outputs probabilities (P(UP))

Why Logistic Regression?

Easy to debug

Stable baseline

Clear feature impact

Common interview expectation

7. Evaluation Metrics

Metrics reported separately for Validation and Test:

Accuracy

Precision / Recall / F1 (for class = UP)

Confusion matrix

Probability distribution of predictions

Naive Baselines Compared

Always predict UP

Always predict DOWN

Random 50%

8. Final Results (Test Set)
Metric	Value
Accuracy	54.4%
Precision (UP)	55.8%
Recall (UP)	52.8%
F1 (UP)	54.3%

Baseline comparison:

Random ≈ 50%

Always UP ≈ 51%

Model beats all naive baselines

Important observation:

Model confidence clusters near 0.5

Small but meaningful high-confidence tail (>0.6)

This indicates the model is better used as a selective signal, not a daily predictor.

9. Leakage Precautions (Explicit)

The following safeguards are enforced:

Time-based split (no shuffle)

Rolling features use only past data

Label uses future data but is never used in features

StandardScaler fit only on training data

Validation and test never influence training

This avoids:

lookahead bias

data leakage

overly optimistic metrics

10. Reproducibility

To reproduce results:

# Stage 6 — Train baseline
python src/train_baseline.py

# Stage 7 — Evaluate baseline
python src/evaluate_baseline.py


Artifacts produced:

models/logreg_baseline.joblib

reports/metrics.json

reports/summary.md

11. Key Takeaways

Even simple models can extract weak but real signal

Accuracy alone is misleading; probability ranking matters

Honest evaluation is more valuable than complex models

This pipeline forms a solid foundation for future upgrades

12. Next Improvements (Planned)

Concrete, non-vague upgrades:

Walk-Forward Validation

Rolling train → test windows

Measure stability across regimes

Threshold-Based Trading

Trade only when P(UP) > 0.6

Optimize precision vs frequency

Multi-Stock Training

Pool multiple symbols

Learn cross-sectional patterns

Feature Expansion

Trend filters

Regime indicators

Relative strength vs index

Deep Learning Upgrade Path

Sequence models (MLP → LSTM/TCN)

Probability calibration

Risk-aware position sizing

13. Status

✅ Baseline pipeline complete
✅ Evaluation honest and reproducible
🔜 Strategy logic & advanced models planned

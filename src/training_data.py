import pandas as pd
from sklearn.preprocessing import StandardScaler


def train_data(df: pd.DataFrame):
    train_start, train_end = "2015-01-01", "2015-12-31"
    val_start, val_end = "2016-01-01", "2016-12-31"
    test_start, test_end = "2017-01-01", "2017-12-31"

    train_mask = (df["Date"] >= train_start) & (df["Date"] <= train_end)
    val_mask = (df["Date"] >= val_start) & (df["Date"] <= val_end)
    test_mask = (df["Date"] >= test_start) & (df["Date"] <= test_end)

    df_train = df.loc[train_mask].copy()
    df_val = df.loc[val_mask].copy()
    df_test = df.loc[test_mask].copy()


    assert len(df_train) > 0 and len(df_val) > 0 and len(df_test) > 0, "One of the splits is empty."

    max_train_date = df_train["Date"].max()
    max_val_date = df_val["Date"].max()
    min_val_date = df_val["Date"].min()
    min_test_date = df_test["Date"].min()

    assert max_train_date < min_val_date, f"Train overlaps Val: {max_train_date} !< {min_val_date}"
    assert max_val_date < min_test_date, f"Val overlap Test: {max_val_date} !< {min_test_date}"

    print("Date split OK:")
    print("Train:", df_train["Date"].min(), "->", df_train["Date"].max(), "rows:", len(df_train))
    print("Val:", df_val["Date"].min(), "->", df_val["Date"].max(), "rows:", len(df_val))
    print("Test:", df_test["Date"].min(), "->", df_test["Date"].max(), "rows:", len(df_test))

    feature_col = [c for c in df.columns if c not in ["Date", "y"]]

    X_train = df_train[feature_col].copy()
    Y_train = df_train["y"].copy()

    X_val = df_val[feature_col].copy()
    Y_val = df_val["y"].copy()

    X_test = df_test[feature_col].copy()
    Y_test = df_test["y"].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.fit_transform(X_test)

    print("Scaler fit only on train. Transformed val/test.")

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_col, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_col, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_col, index=X_test.index)

    print("\nClass balance:")
    print("Train y mean: ", Y_train.mean())
    print("Val y mean: ", Y_val.mean())
    print("Test y mean: ", Y_test.mean())
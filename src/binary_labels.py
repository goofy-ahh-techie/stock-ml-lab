import pandas as pd

def create_binary_nextday_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates next-day return and binary label (up/down).
    """
    new_df = df.copy()

    """
    Compute neyt day forward return
    ret_fwd_1[t] = (Close[t+1] - Close[t]) / Close[t]
    """
    new_df["ret_fwd_1"] = ( new_df["Close"].shift(-1) - new_df["Close"] ) / new_df["Close"]

    """
    Create binary labels
    1 = price went up next day
    0 = price went down next day
    """
    new_df["y"] = ( new_df["ret_fwd_1"] > 0).astype("int8")

    """
    Remove rows with NaN labels (last row + any bad data)
    """
    before = len(new_df)
    new_df = new_df.dropna(subset=["ret_fwd_1", "y"]).reset_index(drop=True)
    dropped = before - len(new_df)
    print(f"[LABEL] Dropped {dropped} rows (last row / bad Close).")

    """ Sanity Checks """
    assert set(new_df["y"].unique()).issubset({0,1}), "Label contains invalid values"
    assert new_df["y"].isna().sum() == 0, "Label contains NaNs"

    """ Print class balance """
    pct_up = new_df["y"].mean() * 100
    print(f"[LABEL] Class balance -> Up: {pct_up: .2f}% | Down: {100 - pct_up: .2f}%")
    new_df.to_csv("data/processed/RELIANCE.NS.labeled.csv", index=False)

    return new_df
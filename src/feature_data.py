import pandas as pd

def featuring_the_data(df: pd.DataFrame):

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["vol_5"] = df["ret_1"].rolling(window=5).std()


    df["vol_chg_1"] = df["Volume"].pct_change(1)
    df["vol_ratio_5"] = df["Volume"] / df["Volume"].rolling(5).mean()

    feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "vol_5",
        "vol_chg_1",
        "vol_ratio_5",
    ]

    final_cols = ["Date"] + feature_cols + ["y"]
    df_final = df[final_cols]

    df_final = df_final.dropna().reset_index(drop=True)

    df_final.to_csv("data/processed/RELIANCE.NS.featured.csv", index=False)

    print("Feature engineering complete.")
    print("Final shape: ", df_final.shape)
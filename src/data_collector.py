import yfinance as yf
import pandas as pd


def clean_daily_data(stock_name: str): 
    """ 
    Basically initially call the yfinance API for getting the data, and save that data in a csv file
    """
    dataF = yf.download(stock_name, start="2015-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    if dataF is None or dataF.empty:
        raise ValueError(f"No data returned for symbol={stock_name}. Check ticker, date range, or network.")
    
    dataF.to_csv("data/raw/" + stock_name + ".csv", index=True)

    """
    Now we will fetch raw data from that csv file, and make it clean and readable. 
    """
    path = "data/raw/" + stock_name + ".csv"
    df = pd.read_csv(path)

    """Will be removing top 3 rows as they were some corrupted data."""
    df = df.iloc[2:]

    """Will be adding column names on top of the data."""
    df.columns = [
        "Date",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    change_price_to_nums(df)
    change_date_type(df)
    drop_missing_ohlcv(df)

    df.to_csv("data/processed/" + stock_name + ".csv", index=False)

def change_price_to_nums(df: pd.DataFrame):
    """
    This will round off the price values to 4 after decimal.
    """
    price_col = [
        "Close",
        "High",
        "Low",
        "Open"
    ]
    df[price_col] = (df[price_col].apply(
        pd.to_numeric, errors="coerce"
    ).round(4))
    return df

def change_date_type(df: pd.DataFrame):
    """Will be updating the date time format to UTC."""
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d") # type: ignore

def drop_missing_ohlcv(df: pd.DataFrame):
    ohlcv_cols = ["Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    before = len(df)
    df = df.dropna(subset=ohlcv_cols)
    after = len(df)

    dropped = before - after
    print(f"[DATA CLEANING] Dropped {dropped} rows with missing OHLCV")

    return df.reset_index(drop=True)
    
clean_daily_data("RELIANCE.NS")
import pandas as pd

TAB_COLS = ["zhvi", "mom_3m", "mom_12m", "vol_12m"]
TEXT_COL = "region_text"

def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["RegionName", "date"])
    g = out.groupby("RegionName", group_keys=False)
    
    out["zhvi_lag_3"] = g["zhvi"].shift(3)
    out["zhvi_lag_12"] = g["zhvi"].shift(12)
    out["mom_3m"] = (out["zhvi"] / out["zhvi_lag_3"]) - 1.0
    out["mom_12m"] = (out["zhvi"] / out["zhvi_lag_12"]) - 1.0
    out["ret_1m"] = g["zhvi"].pct_change()
    out["vol_12m"] = g["ret_1m"].rolling(12).std()
    
    return out.dropna(subset=["mom_3m", "mom_12m", "vol_12m"]).reset_index(drop=True)
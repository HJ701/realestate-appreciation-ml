import argparse
import json
import pandas as pd
from src.utils.config import PATHS
from src.utils.logging import get_logger

log = get_logger(__name__)
ID_COLS = ["RegionID", "RegionName", "RegionType", "StateName"]

def melt_zhvi(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [c for c in df.columns if c not in ID_COLS]
    long = df.melt(id_vars=ID_COLS, value_vars=date_cols, var_name="date", value_name="zhvi")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date", "zhvi"])
    long = long.sort_values(["RegionName", "date"])
    return long

def add_targets(df_long: pd.DataFrame) -> pd.DataFrame:
    g = df_long.groupby("RegionName", group_keys=False)
    df_long["zhvi_fwd_12m"] = g["zhvi"].shift(-12)
    df_long["ret_fwd_12m"] = (df_long["zhvi_fwd_12m"] / df_long["zhvi"]) - 1.0
    df_long["label_appreciate"] = (df_long["ret_fwd_12m"] >= 0).astype(int)
    return df_long.dropna(subset=["ret_fwd_12m"]).reset_index(drop=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zhvi_csv", required=True, help="Path to raw ZHVI CSV")
    p.add_argument("--region_text_json", required=True, help="Path to JSON mapping")
    p.add_argument("--out", default="train_table.parquet", help="Output filename")
    return p.parse_args()

def main():
    args = parse_args()
    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    
    zhvi = pd.read_csv(args.zhvi_csv)
    long = melt_zhvi(zhvi)
    long = add_targets(long)

    text_map = json.loads(open(args.region_text_json, "r", encoding="utf-8").read())
    long["region_text"] = long["RegionName"].map(text_map).fillna("")

    out_path = PATHS.data_processed / args.out
    long.to_parquet(out_path, index=False)
    log.info(f"Wrote dataset: {out_path} rows={len(long)}")

if __name__ == "__main__":
    main()
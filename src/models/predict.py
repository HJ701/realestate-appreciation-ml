import argparse
import pandas as pd
import joblib
from src.utils.config import PATHS
from src.features.build_features import add_tabular_features, TAB_COLS, TEXT_COL

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ridge_combined")
    p.add_argument("--out", default="predictions.csv")
    return p.parse_args()

def main():
    args = parse_args()
    df = add_tabular_features(pd.read_parquet(PATHS.data_processed / "train_table.parquet"))
    X = df[TAB_COLS + [TEXT_COL]].copy()
    
    reg = joblib.load(PATHS.models / f"{args.model}.joblib")
    pred = reg.predict(X)
    
    out = df[["date", "RegionName"]].copy()
    out["pred_ret_fwd_12m"] = pred
    
    PATHS.results.mkdir(parents=True, exist_ok=True)
    out.to_csv(PATHS.results / args.out, index=False)

if __name__ == "__main__":
    main()
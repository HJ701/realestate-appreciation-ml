import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import RocCurveDisplay
from src.utils.config import PATHS
from src.features.build_features import add_tabular_features, TAB_COLS, TEXT_COL

def main():
    df = add_tabular_features(pd.read_parquet(PATHS.data_processed / "train_table.parquet"))
    X = df[TAB_COLS + [TEXT_COL]].copy()
    y = df["label_appreciate"].astype(int)
    
    clf = joblib.load(PATHS.models / "clf_combined_logreg.joblib")
    prob = clf.predict_proba(X)[:, 1]
    
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y, prob)
    plt.title("Appreciation vs Depreciation (AUC)")
    plt.savefig(PATHS.figures / "roc_appreciation.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
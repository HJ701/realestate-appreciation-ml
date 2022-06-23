import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.utils.config import PATHS
from src.utils.logging import get_logger
from src.features.build_features import add_tabular_features, TAB_COLS, TEXT_COL

log = get_logger(__name__)

def load_table():
    return pd.read_parquet(PATHS.data_processed / "train_table.parquet")

def make_preprocessor(use_tab=True, use_text=True):
    transformers = []
    if use_tab:
        transformers.append(("tab", Pipeline([("scaler", StandardScaler())]), TAB_COLS))
    if use_text:
        transformers.append(("txt", TfidfVectorizer(max_features=2000, ngram_range=(1,2)), TEXT_COL))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)

def train_regressors(X_train, y_train):
    models = {
        "ridge_tab": Pipeline([("pre", make_preprocessor(True, False)), ("m", Ridge(alpha=1.0))]),
        "rf_tab": Pipeline([("pre", make_preprocessor(True, False)), ("m", RandomForestRegressor(n_estimators=250, random_state=42))]),
        "gbr_tab": Pipeline([("pre", make_preprocessor(True, False)), ("m", GradientBoostingRegressor(random_state=42))]),
        "ridge_text": Pipeline([("pre", make_preprocessor(False, True)), ("m", Ridge(alpha=1.0))]),
        "ridge_combined": Pipeline([("pre", make_preprocessor(True, True)), ("m", Ridge(alpha=1.0))]),
    }
    for k, pipe in models.items():
        pipe.fit(X_train, y_train)
        log.info(f"fit: {k}")
    return models

def train_classifier(X_train, y_train):
    clf = Pipeline([("pre", make_preprocessor(True, True)), ("m", LogisticRegression(max_iter=400, class_weight="balanced"))])
    clf.fit(X_train, y_train)
    return clf

def main():
    PATHS.models.mkdir(parents=True, exist_ok=True)
    PATHS.results.mkdir(parents=True, exist_ok=True)
    df = add_tabular_features(load_table())
    
    y_reg = df["ret_fwd_12m"].astype(float)
    y_cls = df["label_appreciate"].astype(int)
    X = df[TAB_COLS + [TEXT_COL]].copy()

    X_tr, X_te, y_r_tr, y_r_te, y_c_tr, y_c_te = train_test_split(X, y_reg, y_cls, test_size=0.2, random_state=42, shuffle=True)

    regs = train_regressors(X_tr, y_r_tr)
    leaderboard = []
    for name, pipe in regs.items():
        pred = pipe.predict(X_te)
        mae = float(mean_absolute_error(y_r_te, pred))
        r2 = float(r2_score(y_r_te, pred))
        leaderboard.append({"model": name, "mae": mae, "r2": r2})
        joblib.dump(pipe, PATHS.models / f"{name}.joblib")

    clf = train_classifier(X_tr, y_c_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_c_te, prob))
    joblib.dump(clf, PATHS.models / "clf_combined_logreg.joblib")

    pd.DataFrame(leaderboard).sort_values("mae").to_csv(PATHS.results / "leaderboard_regression.csv", index=False)
    (PATHS.results / "train_meta.json").write_text(json.dumps({"classification_auc": auc}, indent=2))
    log.info(f"Saved results. AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
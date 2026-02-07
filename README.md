# 🏠 Real Estate Value Prediction: ML-Powered Investment Analysis

![Project Banner](data/real_estate_ml.png)

## 👋 About the Project

This repository implements a machine learning pipeline for predicting real estate value appreciation using Zillow Home Value Index (ZHVI) time series data combined with Wikipedia-based regional text features. The project tackles both regression (predicting 12-month percentage returns) and classification (identifying value gain/loss) through a hybrid approach that merges numerical market indicators with textual region characteristics.

Built as a practical exploration of feature engineering, this system demonstrates how combining structured financial data with unstructured text can enhance real estate investment predictions.

## 🎯 What Does It Do?

- **Data Integration**: Automatically downloads ZHVI data and fetches Wikipedia region summaries via API
- **Feature Engineering**: Computes momentum (3-month, 12-month) and volatility indicators from price history
- **Text Processing**: Extracts TF-IDF features (max 2000 features, 1-2 n-grams) from regional descriptions
- **Dual Modeling**: Trains both regression models (Ridge, Random Forest, Gradient Boosting) and classification models (Logistic Regression)
- **Target**: Predicts 12-month forward returns and binary value change direction

## 🛠️ Installation

Set up your environment with the following steps:

```bash
# Clone the repository
git clone https://github.com/username/real-estate-value-prediction.git
cd real-estate-value-prediction

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate

# Install dependencies (including dev tools: ruff, pytest)
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, Pandas ≥2.0, Scikit-learn ≥1.3, Requests, Matplotlib, Numpy

## 🚀 Usage

Follow these steps to run the complete pipeline:

### 1. Download ZHVI Data

```bash
python -m src.data.download_zhvi --url "CSV_URL" --out data/raw/zhvi.csv
```

Downloads the Zillow Home Value Index dataset directly from the provided URL.

### 2. Fetch Regional Text Data

```bash
python -m src.data.fetch_region_text --zhvi data/raw/zhvi.csv --out data/raw/region_text.csv
```

Retrieves Wikipedia summaries for regions in the ZHVI dataset via API.

### 3. Build Training Dataset

```bash
python -m src.data.make_dataset --zhvi data/raw/zhvi.csv --text data/raw/region_text.csv --out data/processed/train.csv
```

Merges raw data, calculates 12-month forward returns, and prepares the final training dataset.

### 4. Train Models

```bash
python -m src.models.train
```

Trains all configured models (Ridge, Random Forest, Gradient Boosting for regression; Logistic Regression for classification). Results are saved to `reports/results/leaderboard_regression.csv`.

### 5. Evaluate Performance

```bash
python -m src.models.evaluate
```

Generates evaluation metrics and visualizations (e.g., ROC curves for classification).

### 6. Make Predictions

```bash
python -m src.models.predict --model ridge_combined --input data/processed/test.csv
```

Applies a trained model to generate predictions on new data.

## 🧠 Model Architecture

The project uses a hybrid feature approach combining tabular and text data:

### Feature Engineering

| Feature Type | Description | Details |
|--------------|-------------|---------|
| **Tabular Features** | Price-based indicators | `mom_3m`: 3-month momentum<br>`mom_12m`: 12-month momentum<br>`vol_12m`: 12-month volatility |
| **Text Features** | Regional characteristics | TF-IDF on Wikipedia summaries<br>Max 2000 features, 1-2 n-grams |

### Models

| Task | Models | Purpose |
|------|--------|---------|
| **Regression** | Ridge, Random Forest Regressor, Gradient Boosting Regressor | Predict 12-month % return |
| **Classification** | Logistic Regression | Predict value gain/loss direction |

## 📊 Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── download_zhvi.py      # Zillow data downloader
│   │   ├── fetch_region_text.py  # Wikipedia API integration
│   │   └── make_dataset.py       # Dataset builder
│   ├── features/
│   │   └── build_features.py     # Momentum & volatility calculation
│   └── models/
│       ├── train.py              # Model training pipeline
│       ├── evaluate.py           # Performance evaluation
│       └── predict.py            # Inference script
├── reports/
│   ├── results/                  # Leaderboards & metrics
│   ├── models/                   # Saved .joblib files
│   └── figures/                  # Plots & visualizations
├── data/                         # Ignored by git (local only)
│   ├── raw/                      # Downloaded ZHVI & text
│   └── processed/                # Training datasets
└── README.md
```

**Note**: The `data/` directory is git-ignored; datasets are downloaded locally.

## 📈 Sample Output

After training, expect performance metrics like:

```
Regression Leaderboard:
Model               RMSE    MAE     R²
ridge_combined      2.34    1.82    0.67
rf_combined         2.51    1.95    0.63
gbm_combined        2.29    1.78    0.69

Classification Report:
              precision    recall  f1-score   support
           0       0.73      0.68      0.70       234
           1       0.75      0.79      0.77       390
```

**Note**: Results vary due to data splits and randomness.

## 🔬 Technical Details

### Data Pipeline

1. **Download**: ZHVI CSV fetched via HTTP request
2. **Text Enrichment**: Wikipedia API calls for region summaries (rate-limited, includes retries)
3. **Feature Calculation**: Rolling windows for momentum/volatility
4. **Target Creation**: Forward 12-month returns calculated as `(price_t+12 / price_t) - 1`

### Modeling Approach

- **Train/Test Split**: 80/20 temporal split (preserves time ordering)
- **Hyperparameters**: Default scikit-learn settings (tune via grid search if needed)
- **Evaluation Metrics**: RMSE, MAE, R² for regression; Precision, Recall, F1 for classification

## 📝 TODO

Future enhancements and improvements:

- [ ] Add cross-validation for more robust evaluation
- [ ] Implement hyperparameter tuning (GridSearchCV/Optuna)
- [ ] Integrate external data sources (interest rates, economic indicators)
- [ ] Build interactive dashboard (Streamlit/Gradio)
- [ ] Add model explainability (SHAP values)
- [ ] Expand text sources beyond Wikipedia
- [ ] Deploy as REST API

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest features via Issues
- Submit pull requests for improvements
- Share dataset enhancements or new data sources

## 📄 License

This project is open source and for educational purposes only. Not intended for real investment decisions—always consult qualified financial advisors before making investment choices. 📊

---

**Disclaimer**: This is an experimental ML project. Real estate investment involves significant risk. Past performance does not guarantee future results.

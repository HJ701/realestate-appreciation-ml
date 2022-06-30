# Real Estate Appreciation ML (Tabular + Text)
Small real-estate investment ML project that predicts:
- **Forward 12-month % change** (regression)
- **Appreciation vs depreciation** (classification)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run
1. `python -m src.data.download_zhvi --url "URL"` 
2. `python -m src.data.fetch_region_text --regions_csv data/raw/zhvi.csv` 
3. `python -m src.data.make_dataset --zhvi_csv data/raw/zhvi.csv --region_text_json data/processed/region_text.json` 
4. `python -m src.models.train` 
5. `python -m src.models.evaluate` 
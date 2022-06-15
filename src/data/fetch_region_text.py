import argparse
import json
from pathlib import Path
import requests
import pandas as pd
from src.utils.config import PATHS
from src.utils.logging import get_logger

log = get_logger(__name__)

def wiki_summary(title: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "realestate-appreciation-ml/1.0"})
    if r.status_code != 200:
        return ""
    j = r.json()
    return (j.get("extract") or "").strip()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regions_csv", required=True, help="CSV with column RegionName")
    p.add_argument("--out", default="region_text.json", help="Output under data/processed/")
    p.add_argument("--limit", type=int, default=200, help="Limit regions (for quick runs)")
    return p.parse_args()

def main():
    args = parse_args()
    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.data_processed / args.out

    df = pd.read_csv(args.regions_csv)
    names = df["RegionName"].dropna().astype(str).unique().tolist()[:args.limit]

    def to_title(region_name: str) -> str:
        base = region_name.split(",")[0].strip()
        return base.replace(" ", "_")

    data = {}
    for n in names:
        title = to_title(n)
        data[n] = wiki_summary(title)
        log.info(f"text: {n} -> {len(data[n])} chars")

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
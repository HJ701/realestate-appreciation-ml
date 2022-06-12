import argparse
from pathlib import Path
import requests
from src.utils.config import PATHS
from src.utils.logging import get_logger

log = get_logger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Direct CSV URL for a Zillow ZHVI table")
    p.add_argument("--out", default="zhvi.csv", help="Output filename under data/raw/")
    return p.parse_args()

def main():
    args = parse_args()
    PATHS.data_raw.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.data_raw / args.out
    
    r = requests.get(args.url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    log.info(f"Downloaded: {out_path}")

if __name__ == "__main__":
    main()
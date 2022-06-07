from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    results: Path = ROOT / "reports" / "results"
    figures: Path = ROOT / "reports" / "figures"
    models: Path = ROOT / "reports" / "models"

PATHS = Paths()
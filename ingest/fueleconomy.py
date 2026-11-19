"""Ingest EPA fuel economy data from fueleconomy.gov.

Downloads the bulk vehicles CSV (zipped), keeps the columns we need,
and saves it as raw Parquet to data/raw/fueleconomy/vehicles.parquet.

No cleaning happens here: the raw layer is a faithful copy of the source.
Normalization and the make/model/year join key are handled in dbt staging.

Run from the repo root:
    python ingest/fueleconomy.py
"""

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"

# Resolve paths relative to the repo root, not the current working directory,
# so the script behaves the same no matter where it is run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "fueleconomy" / "vehicles.parquet"

# The source has ~80 columns; we keep only what downstream models need.
COLUMNS = [
    # identity / join key
    "id",           # EPA's unique row id
    "year",
    "make",
    "model",        # often includes trim/drive, e.g. "F150 Pickup 4WD"
    "baseModel",    # cleaner model name, likely better for joining to listings
    # vehicle attributes
    "VClass",       # EPA size class, e.g. "Compact Cars"
    "drive",
    "trany",        # transmission
    "cylinders",
    "displ",        # engine displacement (liters)
    # fuel
    "fuelType1",    # primary fuel
    "atvType",      # alternative fuel type: Hybrid, EV, Plug-in Hybrid, ...
    # efficiency (MPG)
    "city08",
    "highway08",
    "comb08",
    # annual cost, used in the 5-year TCO
    "fuelCost08",   # EPA's estimated annual fuel cost (USD)
    "barrels08",    # annual petroleum consumption (barrels)
    # electric vehicles
    "combE",        # combined electricity use (kWh / 100 mi)
    "range",        # EV range (miles)
]


def download(url: str) -> bytes:
    """Download the file and return its raw bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()  # fail loudly on 4xx/5xx instead of saving an error page
    return response.content


def load(raw_bytes: bytes) -> pd.DataFrame:
    """Read the CSV inside the zip into a DataFrame with only COLUMNS."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in the zip, found: {zf.namelist()}")
        with zf.open(csv_names[0]) as f:
            # usecols raises a ValueError if any column is missing,
            # so a schema change at the source fails loudly.
            df = pd.read_csv(f, usecols=COLUMNS, low_memory=False)

    # Lineage: record when this snapshot was pulled.
    df["ingested_at"] = datetime.now(timezone.utc)
    return df[COLUMNS + ["ingested_at"]]  # stable column order


def save(df: pd.DataFrame, path: Path) -> None:
    """Write the DataFrame to Parquet, creating folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def main() -> None:
    raw_bytes = download(URL)
    df = load(raw_bytes)
    save(df, OUTPUT_PATH)
    print(
        f"Saved {len(df):,} rows x {df.shape[1]} cols "
        f"(model years {df['year'].min()}-{df['year'].max()}) "
        f"to {OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()

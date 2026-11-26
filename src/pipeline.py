"""
Core data pipeline for the Used Car Valuation & Price Regression project.

Stage 1 (eda-and-cleaning): Data discovery and Pandas/Regex specification cleaning.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

PREFERRED_FILES = [
    "Car details v3.csv",
    "car details v4.csv",
    "CAR DETAILS FROM CAR DEKHO.csv",
    "car data.csv",
]

# Specification definitions: (output_col, candidate_raw_cols, audit_label, sample_label)
SPEC_FIELDS = [
    ("mileage_kmpl", ["mileage"], "mileage_kmpl", "Mileage (kmpl)"),
    ("engine_cc", ["engine", "engine_cc"], "engine_cc", "Engine (cc)"),
    ("power_bhp", ["max_power", "max power", "max_power_bhp"], "power_bhp", "Power (bhp)"),
]


def list_data_files() -> List[Path]:
    """Return available CSV datasets from `data/` (or `archive/` fallback)."""
    base_dir = DATA_DIR if (DATA_DIR.exists() and any(DATA_DIR.glob("*.csv"))) else ARCHIVE_DIR
    if not base_dir.exists():
        return []
    files = [base_dir / name for name in PREFERRED_FILES if (base_dir / name).is_file()]
    seen = {p.name for p in files}
    for path in sorted(base_dir.glob("*.csv")):
        if path.name not in seen:
            files.append(path)
    return files


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name case-insensitively."""
    lookup = {col.strip().lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def _extract_number(series: pd.Series) -> pd.Series:
    """Extract the first positive numeric value from strings like '23.4 kmpl' or '1248 CC'."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = series.astype(str).str.replace(",", "", regex=False)
        vals = pd.to_numeric(cleaned.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    return vals.where(vals > 0, np.nan)


def _extract_torque_nm(series: pd.Series) -> pd.Series:
    """Extract torque numerically via Regex and convert kgm values to Nm (1 kgm = 9.80665 Nm)."""
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        return vals.where(vals > 0, np.nan)

    def _parse(val: Any) -> float:
        if pd.isna(val):
            return np.nan
        text = str(val).replace(",", "")
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return np.nan
        num = float(match.group(1))
        if num <= 0:
            return np.nan
        if re.search(r"\bkgm\b", text, re.IGNORECASE) and not re.search(r"\bnm\b", text, re.IGNORECASE):
            return num * 9.80665
        return num

    return series.apply(_parse)


def _extract_brand_and_name(raw: pd.DataFrame, make_col: Optional[str], model_col: Optional[str], name_col: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    """Standardize vehicle make and model name using Regex whitespace cleanup."""
    if make_col and model_col:
        brand = (
            raw[make_col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace({"Maruti Suzuki": "Maruti"})
        )
        car_name = (brand + " " + raw[model_col].astype(str).str.strip()).str.strip()
        return brand, car_name
    if name_col:
        car_name = raw[name_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        brand = car_name.str.extract(r"^([^\s]+)", expand=False).str.title()
        return brand, car_name
    unknown = pd.Series("Unknown", index=raw.index)
    return unknown, unknown


def clean_specifications(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a dataset CSV and clean vehicle specifications using Pandas and Regex.

    Returns `(clean_df, regex_audit_df, sample_before_after_df)`.
    """
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame(index=raw.index)

    price_col = _find_col(raw, ["selling_price", "price"])
    year_col = _find_col(raw, ["year"])
    km_col = _find_col(raw, ["km_driven", "kms_driven", "kilometer"])
    fuel_col = _find_col(raw, ["fuel", "fuel_type", "fuel type"])
    trans_col = _find_col(raw, ["transmission"])
    owner_col = _find_col(raw, ["owner"])
    name_col = _find_col(raw, ["name", "car_name"])
    make_col = _find_col(raw, ["make"])
    model_col = _find_col(raw, ["model"])
    torque_col = _find_col(raw, ["torque", "max torque", "max_torque_nm"])
    seats_col = _find_col(raw, ["seats", "seating capacity", "seating_capacity"])

    if not price_col or not year_col:
        raise ValueError(f"Expected price and year columns in {csv_path.name}")

    # Price standardized to INR Lakhs (`car data.csv` stores selling_price already in Lakhs)
    raw_price = _extract_number(raw[price_col])
    if price_col.lower() == "selling_price" and raw_price.dropna().median() < 500:
        df["price_lakh"] = raw_price
    else:
        df["price_lakh"] = raw_price / 100_000.0

    df["year"] = pd.to_numeric(raw[year_col], errors="coerce")
    df["km_driven"] = _extract_number(raw[km_col]) if km_col else np.nan
    df["fuel_type"] = raw[fuel_col].astype(str).str.strip().str.title() if fuel_col else "Unknown"
    df["transmission"] = raw[trans_col].astype(str).str.strip().str.title() if trans_col else "Unknown"
    df["brand"], df["car_name"] = _extract_brand_and_name(raw, make_col, model_col, name_col)
    if owner_col:
        df["owner"] = raw[owner_col].astype(str).str.strip()

    audit_rows: List[Dict[str, Any]] = []
    parsed_specs: List[Tuple[str, str, str]] = []
    for out_col, candidates, field_name, label in SPEC_FIELDS:
        src_col = _find_col(raw, candidates)
        if src_col:
            df[out_col] = _extract_number(raw[src_col])
            parsed_specs.append((src_col, out_col, label))
            audit_rows.append(
                {
                    "Field": field_name,
                    "Source": src_col,
                    "Regex": r"(\d+(?:\.\d+)?)",
                    "Parsed": int(df[out_col].notna().sum()),
                    "Raw Non-Null": int(raw[src_col].notna().sum()),
                }
            )

    if torque_col:
        df["torque_nm"] = _extract_torque_nm(raw[torque_col])
        parsed_specs.append((torque_col, "torque_nm", "Torque (Nm)"))
        audit_rows.append(
            {
                "Field": "torque_nm",
                "Source": torque_col,
                "Regex": r"(\d+(?:\.\d+)?) + kgm->Nm",
                "Parsed": int(df["torque_nm"].notna().sum()),
                "Raw Non-Null": int(raw[torque_col].notna().sum()),
            }
        )

    if seats_col:
        df["seats"] = pd.to_numeric(raw[seats_col], errors="coerce")

    # Capture compact before/after sample before dropping duplicate rows
    sample_cols = [c for c in [name_col, make_col] + [s[0] for s in parsed_specs] if c]
    sample_idx = raw.dropna(subset=sample_cols[: min(2, len(sample_cols))]).head(12).index
    before_after = pd.DataFrame(index=sample_idx)
    if name_col:
        before_after["Raw Name"] = raw.loc[sample_idx, name_col]
    before_after["Brand"] = df.loc[sample_idx, "brand"]
    for src_col, clean_col, label in parsed_specs:
        before_after[f"Raw {src_col}"] = raw.loc[sample_idx, src_col]
        before_after[label] = df.loc[sample_idx, clean_col].round(1)

    df = df.dropna(subset=["price_lakh", "year"]).drop_duplicates().reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    return df, pd.DataFrame(audit_rows), before_after.reset_index(drop=True)

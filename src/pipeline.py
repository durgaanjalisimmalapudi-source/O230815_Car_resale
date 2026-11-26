"""
Core data pipeline for the Used Car Valuation & Price Regression project.

Stages:
1. eda-and-cleaning: Data discovery and Pandas/Regex specification cleaning.
2. feature-engineering: Derived metrics, brand median imputation, and feature encoding.
3. modeling-evaluation: OLS regression, multiple regression, and Random Forest confidence estimation.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def _parse_owner_num(value: Any) -> int:
    """Convert heterogeneous owner strings/codes into ordinal numeric ownership count."""
    text = str(value).strip().lower()
    if "test" in text or "unregistered" in text:
        return 0
    for keyword, rank in [("first", 1), ("second", 2), ("third", 3), ("fourth", 4), ("above", 4), ("4 or more", 4)]:
        if keyword in text:
            return rank
    if text.isdigit():
        digit = int(text)
        return digit + 1 if digit <= 3 else digit
    return 1


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `vehicle_age`, `km_10k`, ordinal `owner_num`, and impute missing specs via brand medians."""
    out = df.copy()
    snapshot_year = int(out["year"].max())
    out["vehicle_age"] = (snapshot_year - out["year"]).clip(lower=0)
    out["km_10k"] = out["km_driven"] / 10_000.0
    if "owner" in out.columns:
        out["owner_num"] = out["owner"].apply(_parse_owner_num)

    impute_cols = ["km_driven", "km_10k", "mileage_kmpl", "engine_cc", "power_bhp", "torque_nm", "seats"]
    for col in impute_cols:
        if col in out.columns:
            brand_med = out.groupby("brand")[col].transform("median")
            out[col] = out[col].fillna(brand_med).fillna(out[col].median())
    return out


def prepare_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, Regex-clean, and feature-engineer a dataset CSV from `data/`."""
    clean_df, audit_df, before_after = clean_specifications(csv_path)
    return engineer_features(clean_df), audit_df, before_after


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer for numeric scaling and categorical one-hot encoding."""
    num_candidates = ["vehicle_age", "km_driven", "engine_cc", "power_bhp", "mileage_kmpl", "torque_nm", "seats", "owner_num"]
    cat_candidates = ["brand", "fuel_type", "transmission"]
    num_feats = [c for c in num_candidates if c in df.columns and df[c].notna().any()]
    cat_feats = [c for c in cat_candidates if c in df.columns and df[c].notna().any()]

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
            (
                "cat",
                Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=40, sparse_output=False))]),
                cat_feats,
            ),
        ]
    )
    return pre, num_feats, cat_feats


def run_ols_regression(df: pd.DataFrame, x_col: str) -> Dict[str, Any]:
    """Fit simple OLS linear regression `price_lakh ~ x_col` with a 95% confidence band."""
    work = df[[x_col, "price_lakh"]].dropna()
    work = work[
        (work["price_lakh"] <= work["price_lakh"].quantile(0.98))
        & (work[x_col] <= work[x_col].quantile(0.98))
    ]
    x = work[x_col].to_numpy(dtype=float)
    y = work["price_lakh"].to_numpy(dtype=float)
    res = stats.linregress(x, y)

    y_hat = res.intercept + res.slope * x
    resid = y - y_hat
    n = len(x)
    s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
    t_crit = stats.t.ppf(0.975, df=max(n - 2, 1))

    x_grid = np.linspace(x.min(), x.max(), 80)
    y_grid = res.intercept + res.slope * x_grid
    ci_half = t_crit * s_err * np.sqrt((1.0 / n) + ((x_grid - x.mean()) ** 2 / max(np.sum((x - x.mean()) ** 2), 1e-12)))

    return {
        "n": n,
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue ** 2),
        "pvalue": float(res.pvalue),
        "mae_lakh": float(np.mean(np.abs(resid))),
        "curve": pd.DataFrame(
            {
                "x": x_grid,
                "fit": y_grid,
                "ci_low": np.maximum(0.0, y_grid - ci_half),
                "ci_high": y_grid + ci_half,
            }
        ),
        "sample": work.sample(min(len(work), 1500), random_state=42),
    }


def run_multiple_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Fit `price_lakh ~ vehicle_age + km_10k + specs` with LinearRegression."""
    num_cols = [
        c
        for c in ["vehicle_age", "km_10k", "engine_cc", "power_bhp", "mileage_kmpl", "owner_num"]
        if c in df.columns and df[c].notna().any()
    ]
    cat_cols = [c for c in ["fuel_type", "transmission"] if c in df.columns and df[c].nunique() > 1]

    X = pd.get_dummies(df[num_cols + cat_cols], drop_first=True, dtype=float)
    y = df["price_lakh"].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    labels = {
        "vehicle_age": "Vehicle Age (+1 yr)",
        "km_10k": "Odometer (+10,000 km)",
        "engine_cc": "Engine (+1 cc)",
        "power_bhp": "Power (+1 bhp)",
        "mileage_kmpl": "Mileage (+1 kmpl)",
        "owner_num": "Owner Count (+1)",
    }
    coef_df = pd.DataFrame(
        {
            "Feature": [labels.get(c, c.replace("_", ": ", 1)) for c in X.columns],
            "Price Impact (Lakhs)": model.coef_,
        }
    )
    return coef_df, float(r2_score(y, pred)), float(mean_absolute_error(y, pred))


def train_resale_estimator(df: pd.DataFrame) -> Dict[str, Any]:
    """Train a 100-tree Random Forest estimator; tree percentiles yield confidence bands."""
    pre, num_feats, cat_feats = build_preprocessor(df)
    feats = cat_feats + num_feats

    X = df[feats]
    y = df["price_lakh"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_tr_enc = pre.fit_transform(X_train)
    X_te_enc = pre.transform(X_test)

    rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_tr_enc, y_train)

    te_preds = rf.predict(X_te_enc)
    return {
        "preprocessor": pre,
        "model": rf,
        "features": feats,
        "num_features": num_feats,
        "cat_features": cat_feats,
        "test_r2": float(r2_score(y_test, te_preds)),
        "test_mae": float(mean_absolute_error(y_test, te_preds)),
        "snapshot_year": int(df["year"].max()),
    }


def predict_with_confidence(bundle: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """Predict resale value (INR Lakhs) plus 80% (P10-P90) and 90% (P5-P95) tree intervals."""
    def _eval_batch(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_in = pd.DataFrame([{f: r.get(f, np.nan) for f in bundle["features"]} for r in rows])
        X_enc = bundle["preprocessor"].transform(X_in)
        trees = np.array([t.predict(X_enc) for t in bundle["model"].estimators_])
        return (
            trees.mean(axis=0),
            np.percentile(trees, 10, axis=0),
            np.percentile(trees, 90, axis=0),
            np.percentile(trees, 5, axis=0),
            np.percentile(trees, 95, axis=0),
            trees,
        )

    mean, p10, p90, p05, p95, trees = _eval_batch([row])

    ages = list(range(0, 16))
    a_mean, a10, a90, a05, a95, _ = _eval_batch([{**row, "vehicle_age": a} for a in ages])
    age_curve = pd.DataFrame({"age": ages, "est": a_mean, "p10": a10, "p90": a90, "p05": a05, "p95": a95})

    kms = np.linspace(10_000, 180_000, 18)
    k_mean, k10, k90, k05, k95, _ = _eval_batch([{**row, "km_driven": float(k)} for k in kms])
    km_curve = pd.DataFrame({"km_10k": kms / 10_000.0, "est": k_mean, "p10": k10, "p90": k90, "p05": k05, "p95": k95})

    return {
        "estimate_lakh": float(mean[0]),
        "lo80": float(p10[0]),
        "hi80": float(p90[0]),
        "lo90": float(p05[0]),
        "hi90": float(p95[0]),
        "tree_dist": trees[:, 0],
        "age_curve": age_curve,
        "km_curve": km_curve,
    }

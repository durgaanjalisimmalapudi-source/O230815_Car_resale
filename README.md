# O230815_Car_resale — Used Car Valuation & Price Regression

A clean Python and Streamlit data science project that cleans used-car specifications with Pandas and Regex, analyzes how vehicle age and odometer mileage reduce resale prices via regression, and estimates resale values with confidence charts.

---

## Problem Statement

Used-car listings vary widely in formatting and often embed text units (such as `"190Nm@ 2000rpm"`, `"1248 CC"`, `"Maruti Suzuki"`) alongside missing values and duplicate entries. This project provides a reproducible pipeline to:

1. **Clean Specifications (Pandas & Regex):** Extract numeric engine displacement, power, mileage, and torque (converting `kgm` to `Nm`), standardize brand names, and map ownership tiers cleanly.
2. **Analyze Depreciation via Regression:** Measure how vehicle age and kilometers driven influence selling price using Simple OLS regression (with 95% confidence bands) and Multiple Linear Regression.
3. **Estimate Resale Value with Confidence Bands:** Train a 100-tree Random Forest estimator to deliver a point resale valuation along with 80% and 90% tree-percentile confidence intervals.

---

## Tech Stack

- **Data Processing:** Python 3.12, Pandas, NumPy, Regular Expressions (`re`)
- **Statistics & Modeling:** SciPy (`stats.linregress`), scikit-learn (`LinearRegression`, `RandomForestRegressor`, `ColumnTransformer`)
- **Visualization & Dashboard:** Streamlit, Plotly, Matplotlib
- **Notebooks & Docs:** Executed Jupyter Notebooks (`notebooks/`) and printable PDF report (`docs/Project_Documentation.pdf`)

---

## Repository Structure

```text
O230815_Car_resale/
├── .streamlit/
│   └── config.toml                    # Light-mode Streamlit theme and server config
├── data/
│   └── README.md                      # Dataset source links & schemas (CSV files excluded via .gitignore)
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb      # Executed EDA & Pandas/Regex cleaning audit
│   ├── 02_feature_engineering.ipynb   # Executed age/mileage scaling, imputation & correlation matrix
│   └── 03_modeling_and_evaluation.ipynb # Executed OLS, Multiple Regression & RF benchmarks
├── src/
│   ├── __init__.py
│   ├── pipeline.py                    # Cleaning, feature engineering, regression & valuation pipeline
│   └── app.py                         # 3-tab Streamlit dashboard interface
├── docs/
│   └── Project_Documentation.pdf      # Complete project documentation guide (PDF)
├── .gitignore                         # Ignores environments, cache files, and *.csv datasets
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Dataset Source & `.gitignore` Policy

Following Data Science Git best practices, raw and cleaned `.csv` datasets are excluded from version control via `.gitignore` (`*.csv`).

- **Kaggle Source:** [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- Place the cleaned CSV datasets locally in `data/` (see [data/README.md](data/README.md) for exact filenames and column schemas).

---

## Key Findings (Cleaned Datasets Benchmark)

| Cleaned Dataset | Clean Rows | Age OLS Slope (R²) | Mileage Slope (R²) | Multiple Reg. R² | RF Test R² (MAE) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Car details v3.csv` | 6,908 | `-0.480 L/yr` (`0.343`) | `-0.175 L/10k` (`0.055`) | `0.618` | `0.913` (`₹0.72 L`) |
| `car details v4.csv` | 2,054 | `-1.939 L/yr` (`0.153`) | `-1.506 L/10k` (`0.074`) | `0.673` | `0.673` (`₹3.75 L`) |
| `CAR DETAILS FROM CAR DEKHO.csv` | 3,577 | `-0.457 L/yr` (`0.330`) | `-0.203 L/10k` (`0.063`) | `0.432` | `0.600` (`₹1.59 L`) |
| `car data.csv` | 299 | `-0.365 L/yr` (`0.066`) | `+0.239 L/10k` (`0.020`) | `0.432` | `0.730` (`₹1.49 L`) |

- **Strongest Depreciation Driver:** Vehicle age shows consistent negative slopes (`-₹0.37 L` to `-₹1.94 L` per year depending on market segment), much stronger than odometer mileage alone.
- **High Valuation Accuracy:** Combining age, mileage, brand, fuel type, transmission, and cleaned engine/power/torque specs in `Car details v3.csv` achieves `R² = 0.913` with `MAE = ₹0.72 Lakhs`.

---

## Setup & How to Run

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place cleaned CSVs inside `data/`

Verify `data/` contains `Car details v3.csv`, `car details v4.csv`, `CAR DETAILS FROM CAR DEKHO.csv`, and `car data.csv` (see [data/README.md](data/README.md)).

### 3. Launch the Streamlit app

```bash
.venv/bin/python -m streamlit run src/app.py
```

Open the local URL shown in the terminal (e.g., `http://localhost:8501`).

---

## Documentation

For the printable project report, see [docs/Project_Documentation.pdf](docs/Project_Documentation.pdf).

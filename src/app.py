"""
Used Car Valuation & Price Regression App.

Clean, minimal Streamlit interface for:
1. Resale value estimation with confidence charts
2. Price vs. Age & Mileage regression analysis on `data/`
3. Pandas & Regex specification cleaning audit
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    list_data_files,
    predict_with_confidence,
    prepare_dataset,
    run_multiple_regression,
    run_ols_regression,
    train_resale_estimator,
)


st.set_page_config(page_title="Used Car Valuation & Regression", layout="wide")

# Minimal, non-AI light editorial styling
st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    #MainMenu, footer { visibility: hidden; }
    .val-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 4px solid #1E3A5F;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_prepared(path_str: str, mtime: float):
    return prepare_dataset(Path(path_str))


@st.cache_resource(show_spinner=False)
def _load_model(path_str: str, mtime: float, _df: pd.DataFrame):
    return train_resale_estimator(_df)


def _brand_default(df: pd.DataFrame, brand_med: pd.Series, col: str, lo: float, hi: float) -> float:
    """Return a bounded brand median with global median fallback."""
    val = brand_med.get(col)
    if pd.isna(val):
        val = df[col].median()
    return float(min(hi, max(lo, float(val))))


def _style_fig(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=14, color="#0F172A")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=360,
        margin=dict(l=48, r=20, t=44, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0, font=dict(size=11)),
        xaxis=dict(title=x_title, gridcolor="#F1F5F9"),
        yaxis=dict(title=y_title, gridcolor="#F1F5F9", rangemode="tozero"),
    )
    return fig


def _confidence_chart(
    curve: pd.DataFrame,
    x_col: str,
    x_title: str,
    user_x: float,
    est: float,
    lo90: float,
    hi90: float,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.concat([curve[x_col], curve[x_col][::-1]]),
            y=pd.concat([curve["p95"], curve["p05"][::-1]]),
            fill="toself",
            fillcolor="rgba(30, 58, 95, 0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% Confidence Band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([curve[x_col], curve[x_col][::-1]]),
            y=pd.concat([curve["p90"], curve["p10"][::-1]]),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve[x_col],
            y=curve["est"],
            mode="lines",
            line=dict(color="#1E3A5F", width=2.4),
            name="Estimated Value",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[user_x],
            y=[est],
            mode="markers",
            marker=dict(color="#B45309", size=10),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[max(0.0, hi90 - est)],
                arrayminus=[max(0.0, est - lo90)],
                color="#B45309",
                thickness=2,
            ),
            name="Your Car (90% CI)",
        )
    )
    return _style_fig(fig, title, x_title, "Estimated Price (INR Lakhs)")


def main() -> None:
    files = list_data_files()
    if not files:
        st.error("No CSV datasets found in `data/`. Follow `data/README.md` to download the CarDekho CSV files locally.")
        st.stop()

    with st.sidebar:
        st.header("Dataset")
        chosen_name = st.selectbox("Select CSV dataset", [f.name for f in files], index=0)
        chosen_path = next(f for f in files if f.name == chosen_name)
        mtime = chosen_path.stat().st_mtime
        df, audit_df, before_after_df = _load_prepared(str(chosen_path), mtime)
        model_bundle = _load_model(str(chosen_path), mtime, df)

        st.caption(
            f"Cleaned rows: **{len(df):,}**  \n"
            f"Brands: **{df['brand'].nunique()}**  \n"
            f"Years: **{df['year'].min()} - {df['year'].max()}**  \n"
            f"Model Test R²: **{model_bundle['test_r2']:.3f}** | MAE: **₹{model_bundle['test_mae']:.2f} L**"
        )

    st.title("Used Car Resale Valuation & Price Regression")
    st.caption(f"Cleaned specifications via Pandas & Regex, age/mileage regression, and confidence intervals from `{chosen_path.parent.name}/{chosen_name}`.")

    tab_est, tab_reg, tab_clean = st.tabs(
        ["Resale Estimator", "Price vs. Age & Mileage Regression", "Specification Cleaning (Pandas & Regex)"]
    )

    # --- TAB 1: RESALE ESTIMATOR WITH CONFIDENCE CHARTS ---
    with tab_est:
        col_in, col_out = st.columns([1, 2], gap="large")
        with col_in:
            st.subheader("Vehicle Details")
            brands = sorted(df["brand"].dropna().unique())
            brand = st.selectbox("Brand", brands, index=brands.index("Maruti") if "Maruti" in brands else 0)
            max_yr = model_bundle["snapshot_year"]
            min_yr = int(df["year"].min())
            year = st.number_input("Model Year", min_value=min_yr, max_value=max_yr, value=max(min_yr, max_yr - 4), step=1)
            km_driven = st.number_input("Odometer (km)", min_value=0, max_value=300_000, value=45_000, step=5_000)

            fuels = sorted(df["fuel_type"].dropna().unique())
            fuel_type = st.selectbox("Fuel Type", fuels, index=0)
            trans_opts = sorted(df["transmission"].dropna().unique())
            transmission = st.selectbox("Transmission", trans_opts, index=trans_opts.index("Manual") if "Manual" in trans_opts else 0)

            user_row: Dict[str, Any] = {
                "brand": brand,
                "vehicle_age": float(max(0, max_yr - int(year))),
                "km_driven": float(km_driven),
                "fuel_type": fuel_type,
                "transmission": transmission,
            }

            brand_med = df[df["brand"] == brand].median(numeric_only=True)
            if "owner_num" in model_bundle["num_features"]:
                owner_choice = st.selectbox("Ownership", ["First Owner (1)", "Second Owner (2)", "Third Owner (3)", "Fourth+ Owner (4)"], index=0)
                user_row["owner_num"] = float(int(owner_choice.split("(")[1][0]))
            if "engine_cc" in model_bundle["num_features"]:
                def_eng = int(round(_brand_default(df, brand_med, "engine_cc", 600.0, 6000.0)))
                user_row["engine_cc"] = float(st.number_input("Engine (cc)", 600, 6000, def_eng, 50))
            if "power_bhp" in model_bundle["num_features"]:
                def_pwr = round(_brand_default(df, brand_med, "power_bhp", 30.0, 600.0), 1)
                user_row["power_bhp"] = float(st.number_input("Max Power (bhp)", 30.0, 600.0, def_pwr, 2.0))
            if "mileage_kmpl" in model_bundle["num_features"]:
                def_mil = round(_brand_default(df, brand_med, "mileage_kmpl", 5.0, 40.0), 1)
                user_row["mileage_kmpl"] = float(st.number_input("Fuel Economy (kmpl)", 5.0, 40.0, def_mil, 0.5))
            if "torque_nm" in model_bundle["num_features"]:
                user_row["torque_nm"] = _brand_default(df, brand_med, "torque_nm", 20.0, 1500.0)
            if "seats" in model_bundle["num_features"]:
                user_row["seats"] = _brand_default(df, brand_med, "seats", 2.0, 12.0)

        res = predict_with_confidence(model_bundle, user_row)

        with col_out:
            st.markdown(
                f"""
                <div class="val-box">
                    <div style="font-size:0.8rem;color:#475569;text-transform:uppercase;font-weight:600;">Estimated Resale Value ({brand}, {int(year)})</div>
                    <div style="font-size:2rem;font-weight:700;color:#0F172A;margin:0.15rem 0;">₹{res['estimate_lakh']:.2f} Lakhs <span style="font-size:1.05rem;color:#475569;">(₹{res['estimate_lakh'] * 100_000:,.0f})</span></div>
                    <div style="font-size:0.9rem;color:#334155;">
                        <strong>80% Confidence Range:</strong> ₹{res['lo80']:.2f} L – ₹{res['hi80']:.2f} L &nbsp;|&nbsp;
                        <strong>90% Confidence Range:</strong> ₹{res['lo90']:.2f} L – ₹{res['hi90']:.2f} L
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Point Estimate", f"₹{res['estimate_lakh']:.2f} L")
            m2.metric("80% Interval", f"₹{res['lo80']:.2f}–{res['hi80']:.2f} L")
            m3.metric("Model Test R² / MAE", f"{model_bundle['test_r2']:.3f} / ₹{model_bundle['test_mae']:.2f} L")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    _confidence_chart(res["age_curve"], "age", "Vehicle Age (Years)", user_row["vehicle_age"], res["estimate_lakh"], res["lo90"], res["hi90"], "Resale Value vs. Age (Confidence Bands)"),
                    width="stretch",
                )
            with c2:
                st.plotly_chart(
                    _confidence_chart(res["km_curve"], "km_10k", "Odometer (10,000 km)", user_row["km_driven"] / 10_000.0, res["estimate_lakh"], res["lo90"], res["hi90"], "Resale Value vs. Mileage (Confidence Bands)"),
                    width="stretch",
                )

            fig_hist = go.Figure()
            fig_hist.add_vrect(x0=res["lo90"], x1=res["hi90"], fillcolor="rgba(30,58,95,0.10)", line_width=0, annotation_text="90% CI")
            fig_hist.add_vrect(x0=res["lo80"], x1=res["hi80"], fillcolor="rgba(37,99,235,0.18)", line_width=0, annotation_text="80% CI")
            fig_hist.add_trace(go.Histogram(x=res["tree_dist"], nbinsx=20, marker_color="#1E3A5F", name="Tree Estimates"))
            fig_hist.add_vline(x=res["estimate_lakh"], line_color="#B45309", line_width=2)
            st.plotly_chart(_style_fig(fig_hist, "Prediction Uncertainty Across 100 Regression Trees", "Estimated Price (INR Lakhs)", "Tree Count"), width="stretch")

    # --- TAB 2: PRICE VS. AGE & MILEAGE REGRESSION ---
    with tab_reg:
        age_reg = run_ols_regression(df, "vehicle_age")
        km_reg = run_ols_regression(df, "km_10k")
        coef_df, mv_r2, mv_mae = run_multiple_regression(df)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Age OLS Slope", f"{age_reg['slope']:+.2f} L / yr", f"R² = {age_reg['r2']:.3f}")
        k2.metric("Mileage OLS Slope", f"{km_reg['slope']:+.2f} L / 10k km", f"R² = {km_reg['r2']:.3f}")
        k3.metric("Multiple Reg. R²", f"{mv_r2:.3f}", f"MAE: ₹{mv_mae:.2f} L")
        k4.metric("Observations (Core 98%)", f"{age_reg['n']:,}", f"Snapshot: {model_bundle['snapshot_year']}")

        r1, r2 = st.columns(2)
        for col, reg, x_col, x_lbl, ttl in [
            (r1, age_reg, "vehicle_age", "Vehicle Age (Years)", f"Price vs. Age (Slope: {age_reg['slope']:+.2f} L/yr, R²={age_reg['r2']:.3f})"),
            (r2, km_reg, "km_10k", "Odometer (10,000 km)", f"Price vs. Mileage (Slope: {km_reg['slope']:+.2f} L/10k km, R²={km_reg['r2']:.3f})"),
        ]:
            with col:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=reg["sample"][x_col], y=reg["sample"]["price_lakh"], mode="markers", marker=dict(color="rgba(100,116,139,0.28)", size=5), name="Listings"))
                c = reg["curve"]
                fig.add_trace(go.Scatter(x=pd.concat([c["x"], c["x"][::-1]]), y=pd.concat([c["ci_high"], c["ci_low"][::-1]]), fill="toself", fillcolor="rgba(30,58,95,0.14)", line=dict(color="rgba(0,0,0,0)"), name="95% Mean CI", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=c["x"], y=c["fit"], mode="lines", line=dict(color="#1E3A5F", width=2.4), name="OLS Regression"))
                st.plotly_chart(_style_fig(fig, ttl, x_lbl, "Selling Price (INR Lakhs)"), width="stretch")

        st.subheader("Multiple Linear Regression Coefficients")
        st.caption(f"Jointly models Price (INR Lakhs) on Age, Odometer, and cleaned vehicle specifications (R² = {mv_r2:.3f}, MAE = ₹{mv_mae:.2f} L).")
        coef_disp = coef_df.copy()
        coef_disp["Price Impact (Lakhs)"] = coef_disp["Price Impact (Lakhs)"].round(3)
        st.dataframe(coef_disp, width="stretch", hide_index=True)

    # --- TAB 3: PANDAS & REGEX CLEANING ---
    with tab_clean:
        st.subheader("Regex Specification Extraction Summary")
        if not audit_df.empty:
            st.dataframe(audit_df, width="stretch", hide_index=True)
        else:
            st.info("This CSV already stores numeric/categorical columns directly without unit strings.")

        if not before_after_df.empty:
            st.subheader("Raw vs. Cleaned Sample Rows")
            st.dataframe(before_after_df, width="stretch", hide_index=True)

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df.head(50), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

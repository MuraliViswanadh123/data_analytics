"""
E-Commerce Sales Analytics — Streamlit dashboard.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

The app reads ecommerce_sales.csv from the same folder as this file. To analyse
different data, replace that CSV with your own file using the same columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import streamlit as st

# --------------------------------------------------------------------------- #
# Palette — kept consistent with the notebook / report
INK, SLATE, TEAL, AMBER, GRID = "#1B2430", "#2B5673", "#35786A", "#C56B3E", "#E2E5EA"
CSV_PATH = Path(__file__).with_name("ecommerce_sales.csv")


# --------------------------------------------------------------------------- #
# Data: load the CSV -> clean it. Cleaning is a plain function (easy to test);
# caching is applied by the wrapper get_data().
# --------------------------------------------------------------------------- #


REQUIRED_COLS = ["order_date", "customer_id", "region", "product_category",
                 "product_name", "quantity", "unit_price"]


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse a column to real datetimes whatever text format it arrived in."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    txt = s.astype("string").str.strip()

    # format="mixed" handles genuinely inconsistent rows (pandas >= 2.0);
    # fall back to plain inference on older pandas.
    try:
        best = pd.to_datetime(txt, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        best = pd.to_datetime(txt, errors="coerce")

    # Retry day-first (e.g. 15/03/2024) and keep whichever parses more rows.
    try:
        alt = pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=True)
    except (TypeError, ValueError):
        alt = pd.to_datetime(txt, errors="coerce", dayfirst=True)
    if alt.notna().sum() > best.notna().sum():
        best = alt
    return best


def clean_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Dedupe, standardise, parse dates, coerce numerics, engineer features."""
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise KeyError(
            f"CSV is missing required column(s): {missing}. "
            f"Found: {list(raw.columns)}")

    clean = raw.drop_duplicates().copy()

    # --- dates: convert FIRST, then use .dt -------------------------------
    clean["order_date"] = parse_dates(clean["order_date"])
    if clean["order_date"].isna().all():
        sample = raw["order_date"].dropna().astype(str).head(3).tolist()
        raise ValueError(
            f"Could not parse any values in 'order_date'. Sample: {sample}")
    clean = clean.dropna(subset=["order_date"])
    clean["order_month"] = clean["order_date"].dt.to_period("M").dt.to_timestamp()

    # --- text ---------------------------------------------------------------
    clean["product_category"] = (clean["product_category"].astype("string")
                                 .str.strip().str.title())
    clean["region"] = clean["region"].astype("string").fillna("Unknown")

    # --- numerics: tolerate "$12.50" / "1,234" style values ------------------
    for col in ["quantity", "unit_price", "discount"]:
        if col not in clean.columns:
            clean[col] = 0.0 if col == "discount" else np.nan
        if not pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = pd.to_numeric(
                clean[col].astype("string").str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce")
    clean["discount"] = clean["discount"].fillna(0.0)

    clean = clean[clean["quantity"] > 0]
    clean = clean.dropna(subset=["unit_price"])

    # --- engineered features -------------------------------------------------
    clean["revenue"] = (clean["quantity"] * clean["unit_price"]
                        * (1 - clean["discount"])).round(2)
    clean = clean.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    clean["customer_type"] = np.where(
        clean.groupby("customer_id").cumcount() == 0, "New", "Returning")
    return clean


@st.cache_data(show_spinner=False)
def get_data():
    raw = pd.read_csv(CSV_PATH)
    return raw, clean_df(raw)


# --------------------------------------------------------------------------- #
# Chart helpers — each returns a styled Matplotlib figure (no Streamlit calls)
# --------------------------------------------------------------------------- #
matplotlib.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "figure.autolayout": True,
})


def _k(x, _pos):          # $k axis formatter
    return f"${x/1000:.0f}k"


def _d(x, _pos):          # whole-dollar axis formatter
    return f"${x:,.0f}"


def _fig(w=6.4, h=3.6):
    """A canvas-backed figure — no pyplot global state, safe in a server."""
    fig = Figure(figsize=(w, h))
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


def fig_monthly(df):
    m = df.groupby("order_month")["revenue"].sum()
    fig, ax = _fig(11, 4.0)
    ax.plot(m.index, m.values, color=SLATE, lw=2.4, marker="o", ms=5)
    ax.fill_between(m.index, m.values, color=SLATE, alpha=0.08)
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.margins(x=0.02)
    return fig


def fig_category(df):
    c = df.groupby("product_category")["revenue"].sum().sort_values()
    fig, ax = _fig()
    ax.barh(c.index, c.values, color=SLATE)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.margins(x=0.14)
    for i, v in enumerate(c.values):
        ax.text(v, i, f"  ${v/1000:.0f}k", va="center", fontsize=9)
    ax.grid(axis="y", visible=False)
    return fig


def fig_region(df):
    r = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    fig, ax = _fig()
    ax.bar(r.index, r.values,
           color=[AMBER if x == "Unknown" else SLATE for x in r.index])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.grid(axis="x", visible=False)
    return fig


def fig_order_value(df):
    fig, ax = _fig()
    ax.hist(df["revenue"], bins=40, color=SLATE, alpha=0.85, edgecolor="white")
    med = float(df["revenue"].median())
    ax.axvline(med, color=AMBER, lw=2, ls="--", label=f"Median ${med:,.0f}")
    ax.set_xlabel("Order revenue")
    ax.set_ylabel("Orders")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    return fig


def fig_top_products(df):
    t = (df.groupby("product_name")["revenue"].sum()
         .sort_values(ascending=False).head(10).sort_values())
    fig, ax = _fig(6.4, 4.4)
    ax.barh(t.index, t.values, color=TEAL)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.grid(axis="y", visible=False)
    return fig


def figs_new_returning(df):
    g = df.groupby("customer_type").agg(revenue=("revenue", "sum"),
                                        orders=("order_id", "count"))
    g["aov"] = g["revenue"] / g["orders"]
    g = g.reindex(["New", "Returning"]).dropna()
    frev, arev = _fig(5.2, 3.4)
    arev.bar(g.index, g["revenue"], color=[SLATE, TEAL])
    arev.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    arev.grid(axis="x", visible=False)
    faov, aaov = _fig(5.2, 3.4)
    aaov.bar(g.index, g["aov"], color=[SLATE, TEAL])
    aaov.yaxis.set_major_formatter(mticker.FuncFormatter(_d))
    aaov.grid(axis="x", visible=False)
    return frev, faov


def fig_rfm(df):
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    cust = df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_id", "count"),
        monetary=("revenue", "sum")).reset_index()
    fig, ax = _fig(11, 4.6)
    sc = ax.scatter(cust["frequency"], cust["monetary"], c=cust["recency"],
                    cmap="viridis_r", s=26, alpha=0.8,
                    edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Number of orders")
    ax.set_ylabel("Total spend")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_d))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Recency (days)")
    return fig


def compute_kpis(df):
    return {
        "revenue": float(df["revenue"].sum()),
        "orders": int(len(df)),
        "customers": int(df["customer_id"].nunique()),
        "aov": float(df["revenue"].mean()) if len(df) else 0.0,
        "repeat_rate": float((df.groupby("customer_id").size() > 1).mean()) if len(df) else 0.0,
    }


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="E-Commerce Sales Analytics",
                       page_icon="📊", layout="wide")

    st.markdown("""
        <style>
          .block-container {padding-top: 2.2rem; max-width: 1200px;}
          h1 .accent {color: #2B5673;}
          [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
          [data-testid="stSidebar"] h2 {font-size: 1rem; letter-spacing:.04em;}
        </style>
    """, unsafe_allow_html=True)

    if not CSV_PATH.exists():
        st.error(
            f"Couldn't find **{CSV_PATH.name}** in this folder. "
            "Put the dataset next to app.py and rerun.")
        st.stop()
    try:
        raw, data = get_data()
    except (KeyError, ValueError) as e:
        st.error(f"Couldn't read the dataset: {e}")
        st.stop()
    if data.empty:
        st.error("No usable rows after cleaning — check the CSV's contents.")
        st.stop()

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")
    dmin, dmax = data["order_date"].min().date(), data["order_date"].max().date()
    dr = st.sidebar.date_input("Date range", (dmin, dmax),
                               min_value=dmin, max_value=dmax)
    start, end = (dr if isinstance(dr, tuple) and len(dr) == 2 else (dmin, dmax))

    cats = sorted(data["product_category"].unique())
    regs = sorted(data["region"].unique())
    sel_cats = st.sidebar.multiselect("Category", cats, default=cats)
    sel_regs = st.sidebar.multiselect("Region", regs, default=regs)
    sel_type = st.sidebar.multiselect("Customer type", ["New", "Returning"],
                                      default=["New", "Returning"])

    st.sidebar.divider()
    st.sidebar.caption(
        "Reading `ecommerce_sales.csv` from this folder. Replace it with your own "
        "file (same columns) to analyse different data.")

    # ---- Apply filters ----
    d = data[
        (data["order_date"].dt.date >= start) & (data["order_date"].dt.date <= end)
        & data["product_category"].isin(sel_cats)
        & data["region"].isin(sel_regs)
        & data["customer_type"].isin(sel_type)
    ]

    # ---- Header ----
    st.markdown("<h1>E-Commerce Sales <span class='accent'>Analytics</span></h1>",
                unsafe_allow_html=True)
    st.caption("Interactive dashboard · filter in the sidebar to slice every chart below.")

    if d.empty:
        st.warning("No orders match the current filters. Widen the selection in the sidebar.")
        st.stop()

    with st.expander("Data cleaning summary"):
        st.write(
            f"Raw rows: **{len(raw):,}** → cleaned rows: **{len(data):,}** "
            f"({len(raw) - len(data):,} removed). Steps: dropped duplicate rows, "
            "standardised category labels, parsed dates, removed invalid quantities, "
            "dropped rows missing a price, flagged missing regions as *Unknown*, and "
            "derived `revenue` plus a New/Returning flag from each customer's history.")

    # ---- KPIs ----
    k = compute_kpis(d)
    cols = st.columns(5)
    cards = [
        ("Total revenue", f"${k['revenue']:,.0f}"),
        ("Orders", f"{k['orders']:,}"),
        ("Unique customers", f"{k['customers']:,}"),
        ("Avg order value", f"${k['aov']:,.0f}"),
        ("Repeat-purchase rate", f"{k['repeat_rate']*100:.0f}%"),
    ]
    for col, (label, value) in zip(cols, cards):
        with col.container(border=True):
            st.metric(label, value)

    st.divider()

    # ---- Revenue over time ----
    st.subheader("Revenue over time")
    st.pyplot(fig_monthly(d))

    # ---- Category / Region ----
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by category")
        st.pyplot(fig_category(d))
    with c2:
        st.subheader("Revenue by region")
        st.pyplot(fig_region(d))

    # ---- Order value / Top products ----
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Order value distribution")
        st.pyplot(fig_order_value(d))
    with c4:
        st.subheader("Top products")
        st.pyplot(fig_top_products(d))

    # ---- New vs returning ----
    st.subheader("New vs. returning customers")
    st.caption("Order sizes are similar; returning customers contribute more revenue by buying more often.")
    rev_fig, aov_fig = figs_new_returning(d)
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Revenue by customer type**")
        st.pyplot(rev_fig)
    with c6:
        st.markdown("**Average order value**")
        st.pyplot(aov_fig)

    # ---- RFM ----
    st.subheader("Customer value · RFM")
    st.caption("Each point is a customer, by how often they buy and how much they spend, shaded by recency.")
    st.pyplot(fig_rfm(d))

    st.divider()
    st.caption("Built with Streamlit · same analysis as ecommerce_analytics.ipynb.")


if __name__ == "__main__":
    main()
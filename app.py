import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# --------------------------------------------------------------
# 1) Streamlit Page Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="Ultimate Retail Sales Explorer",
    layout="wide"
)

st.title("Store Sales - The Ultimate Data Exploration & Time Series Tool")
st.caption("**Dataset:** Kaggle’s Store Sales - Time Series Forecasting")

# --------------------------------------------------------------
# 2) Load Main Data (train.csv)
# --------------------------------------------------------------
@st.cache_data
def load_train_data():
    df_train = pd.read_csv("train.csv", parse_dates=["date"])
    return df_train

try:
    df = load_train_data()
except FileNotFoundError:
    st.error("Could not find `train.csv` in the current directory. Please place it next to `app.py`.")
    st.stop()

st.write(f"**Total Rows:** {len(df):,} | **Columns:** {df.shape[1]}")
st.write(
    f"**Date Range:** {df['date'].min().date()} → {df['date'].max().date()} | "
    f"**Stores:** {df['store_nbr'].nunique()} | **Families:** {df['family'].nunique()}"
)

# Preprocessing for time series
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.day_name()  # Day of week
df["week"] = df["date"].dt.isocalendar().week  # Week number

# --------------------------------------------------------------
# 3) Load Extra Datasets
# --------------------------------------------------------------
@st.cache_data
def load_stores():
    return pd.read_csv("stores.csv")

@st.cache_data
def load_oil():
    return pd.read_csv("oil.csv", parse_dates=["date"])

@st.cache_data
def load_holidays():
    return pd.read_csv("holidays_events.csv", parse_dates=["date"])

# Try loading each file, but if not found, show a warning.
try:
    df_stores = load_stores()
except FileNotFoundError:
    st.warning("`stores.csv` not found. Store metadata will not be available.")
    df_stores = None

try:
    df_oil = load_oil()
except FileNotFoundError:
    st.warning("`oil.csv` not found. Oil price data will not be available.")
    df_oil = None

try:
    df_holidays = load_holidays()
except FileNotFoundError:
    st.warning("`holidays_events.csv` not found. Holiday data will not be available.")
    df_holidays = None

# --------------------------------------------------------------
# 4) Sidebar Filters for Core Data
# --------------------------------------------------------------
st.sidebar.header("Filter Options")

all_stores = sorted(df["store_nbr"].unique())
all_families = sorted(df["family"].unique())

selected_stores = st.sidebar.multiselect("Select Store(s)", all_stores, default=all_stores[:5])
selected_families = st.sidebar.multiselect("Select Product Family(ies)", all_families, default=["GROCERY I"])

min_date, max_date = df["date"].min(), df["date"].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date],
                                             min_value=min_date, max_value=max_date)

mask = (
    df["store_nbr"].isin(selected_stores) &
    df["family"].isin(selected_families) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
)
df_filtered = df.loc[mask].copy()
st.write(f"**Filtered Rows:** {len(df_filtered):,}")

if df_filtered.empty:
    st.warning("No data matches these filters. Adjust your selections.")
    st.stop()

# --------------------------------------------------------------
# 5) Tabs for Analysis
# --------------------------------------------------------------
tab_overview, tab_eda, tab_trends, tab_external, tab_protips = st.tabs(
    ["Overview", "Advanced EDA", "Trends & Seasonality", "External Factors", "Pro Tips"]
)

# --------------------------------------------------------------
# Tab 1: Overview
# --------------------------------------------------------------
with tab_overview:
    st.subheader("Overview of Filtered Data")
    with st.expander("Preview Data Sample"):
        st.write(df_filtered.head(20))
    st.markdown("""
    **Instructions:** Use the sidebar to select stores, product families, and a date range.
    The following charts show daily sales and the effect of promotions.
    """)

    # Daily sales line chart
    st.subheader("Daily Total Sales")
    daily_sales = df_filtered.groupby("date")["sales"].sum().reset_index().sort_values("date")
    fig_line = px.line(daily_sales, x="date", y="sales", title="Total Sales Over Time")
    st.plotly_chart(fig_line, use_container_width=True)

    # Scatter plot: onpromotion vs. sales
    st.subheader("Promotions Effect: Sales vs. OnPromotion")
    sample_size = min(10000, len(df_filtered))
    scatter_df = df_filtered.sample(sample_size, random_state=42)
    fig_scatter = px.scatter(scatter_df, x="onpromotion", y="sales", opacity=0.6,
                             title="onpromotion vs. Sales (Sample)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------------------
# Tab 2: Advanced EDA
# --------------------------------------------------------------
with tab_eda:
    st.subheader("Advanced Exploratory Data Analysis")
    
    # (A) Average Sales by Store & Family
    st.markdown("#### Average Sales by Store & Family")
    avg_by_sf = df_filtered.groupby(["store_nbr", "family"])["sales"].mean().reset_index()
    fig_sf = px.bar(avg_by_sf, x="store_nbr", y="sales", color="family", barmode="group",
                    title="Average Sales by Store & Family")
    st.plotly_chart(fig_sf, use_container_width=True)
    
    # (B) Rolling Weekly Average Sales
    st.markdown("#### Rolling Weekly Average Sales")
    daily_agg = df_filtered.groupby("date")["sales"].sum().reset_index().sort_values("date")
    daily_agg["rolling_7"] = daily_agg["sales"].rolling(window=7, min_periods=1).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["sales"],
                                  mode="lines", name="Daily Sales"))
    fig_roll.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["rolling_7"],
                                  mode="lines", name="7-day Rolling Avg"))
    fig_roll.update_layout(title="Daily Sales with 7-day Rolling Average",
                           xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(fig_roll, use_container_width=True)
    
    # (C) Distribution of Sales
    st.markdown("#### Distribution of Sales")
    fig_hist = px.histogram(df_filtered, x="sales", nbins=50, title="Sales Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------------------------------------------
# Tab 3: Trends & Seasonality
# --------------------------------------------------------------
with tab_trends:
    st.subheader("Trends & Seasonality Analysis")
    
    # (A) Day-of-Week Patterns
    st.markdown("#### Average Sales by Day of Week")
    dow_group = df_filtered.groupby("dow")["sales"].mean().reset_index()
    ordered_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_group["dow"] = pd.Categorical(dow_group["dow"], categories=ordered_days, ordered=True)
    dow_group.sort_values("dow", inplace=True)
    fig_dow = px.bar(dow_group, x="dow", y="sales", title="Avg Sales by Day of Week", color="dow")
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # (B) Monthly Patterns
    st.markdown("#### Monthly Patterns")
    mon_group = df_filtered.groupby("month")["sales"].mean().reset_index()
    fig_mon = px.line(mon_group, x="month", y="sales", title="Avg Sales by Month", markers=True)
    fig_mon.update_xaxes(dtick=1)
    st.plotly_chart(fig_mon, use_container_width=True)
    
    # (C) Yearly Trends
    st.markdown("#### Yearly Trends")
    year_group = df_filtered.groupby("year")["sales"].mean().reset_index()
    fig_year = px.bar(year_group, x="year", y="sales", title="Avg Sales by Year", color="year")
    st.plotly_chart(fig_year, use_container_width=True)
    
    # (D) Advanced Decomposition
    st.markdown("#### Advanced Time Series Decomposition")
    st.markdown("Decomposing daily sales into trend, seasonal, and residual components (using a weekly period).")
    daily_ts = daily_agg.set_index("date").asfreq("D").fillna(0)
    decomposition = seasonal_decompose(daily_ts["sales"], model="additive", period=7)
    fig_decomp, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    axes[0].set_title("Decomposition (Additive, period=7)")
    axes[0].plot(decomposition.observed, label="Observed")
    axes[0].legend(loc="upper left")
    axes[1].plot(decomposition.trend, color="red", label="Trend")
    axes[1].legend(loc="upper left")
    axes[2].plot(decomposition.seasonal, color="green", label="Seasonal")
    axes[2].legend(loc="upper left")
    axes[3].plot(decomposition.resid, color="purple", label="Residual")
    axes[3].legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_decomp)
    st.markdown("Interpretation: Trend shows the long-term movement; seasonal shows the repeating pattern (weekly here); residual is what remains.")

# --------------------------------------------------------------
# Tab 4: External Factors (No Earthquake Graph)
# --------------------------------------------------------------
with tab_external:
    st.subheader("External Factors Analysis")
    
    # A) Stores Metadata (stores.csv)
    if df_stores is not None:
        st.markdown("#### Stores Metadata: Sales by Cluster")
        df_with_store = pd.merge(df_filtered, df_stores, on="store_nbr", how="left")
        cluster_group = df_with_store.groupby("cluster")["sales"].mean().reset_index()
        fig_cluster = px.bar(cluster_group, x="cluster", y="sales",
                             title="Average Sales by Store Cluster",
                             labels={"sales": "Avg Sales", "cluster": "Store Cluster"})
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Store metadata not available.")
    
    # B) Oil Prices (oil.csv)
    if df_oil is not None:
        st.markdown("#### Oil Prices vs. Daily Sales")
        df_oil_filtered = df_oil.copy()
        df_oil_filtered["date"] = pd.to_datetime(df_oil_filtered["date"])
        daily_sales_oil = pd.merge(daily_sales, df_oil_filtered, on="date", how="left")
        fig_oil_line = px.line(daily_sales_oil, x="date", y="dcoilwtico",
                               title="Daily Oil Prices Over Time",
                               labels={"dcoilwtico": "Oil Price (USD)"})
        st.plotly_chart(fig_oil_line, use_container_width=True)
        
        fig_oil_scatter = px.scatter(daily_sales_oil, x="dcoilwtico", y="sales",
                                     title="Oil Price vs. Daily Sales",
                                     labels={"dcoilwtico": "Oil Price (USD)", "sales": "Daily Sales"})
        st.plotly_chart(fig_oil_scatter, use_container_width=True)
    else:
        st.info("Oil price data not available.")
    
    # C) Holidays & Events (holidays_events.csv)
    if df_holidays is not None:
        st.markdown("#### Effect of Holidays on Sales")
        df_holidays_clean = df_holidays[~df_holidays["type"].isin(["Transfer", "Bridge"])]
        df_holidays_clean["is_holiday"] = 1
        daily_sales["date_only"] = daily_sales["date"].dt.date
        holidays_indicator = df_holidays_clean[["date", "type"]].drop_duplicates()
        holidays_indicator["date_only"] = holidays_indicator["date"].dt.date
        daily_sales_holiday = pd.merge(daily_sales, holidays_indicator[["date_only"]], on="date_only", how="left")
        daily_sales_holiday["is_holiday"] = daily_sales_holiday["date_only"].apply(lambda x: 1 if x in holidays_indicator["date_only"].values else 0)
        daily_sales_holiday["Holiday"] = daily_sales_holiday["is_holiday"].map({1: "Holiday", 0: "Non-Holiday"})
        fig_holiday = px.box(daily_sales_holiday, x="Holiday", y="sales",
                             title="Distribution of Daily Sales: Holiday vs. Non-Holiday")
        st.plotly_chart(fig_holiday, use_container_width=True)
    else:
        st.info("Holiday and event data not available.")

    # ----------------------------------------------------------
    # (D) Wages Payment Effect
    # ----------------------------------------------------------
    st.markdown("#### (D) Wages Payment Effect on Sales")
    df_filtered["day"] = df_filtered["date"].dt.day
    df_filtered["last_day_of_month"] = df_filtered["date"].dt.days_in_month
    df_filtered["is_wage_day"] = df_filtered.apply(
        lambda row: 1 if (row["day"] == 15 or row["day"] == row["last_day_of_month"]) else 0,
        axis=1
    )
    df_filtered["Wage Payment Day"] = df_filtered["is_wage_day"].map({1: "Wage Day", 0: "Non-Wage Day"})
    fig_wage = px.box(df_filtered, x="Wage Payment Day", y="sales",
                      title="Sales Distribution on Wage Days vs. Non-Wage Days",
                      color="Wage Payment Day")
    st.plotly_chart(fig_wage, use_container_width=True)
    st.markdown("""
    *Public sector wages in Ecuador are paid on **the 15th** and **the last day** of each month.
    This chart checks if there's a noticeable sales boost on these 'wage days.'*
    """)


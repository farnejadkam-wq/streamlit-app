import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil import easter
# For the model
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------------
# 1) Streamlit Page Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Explorer",
    layout="wide"
)

st.title("Store Sales - Data Exploration & Time Series Tool")
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

# --------------------------------------------------------------
# Handle Earthquake Anomaly by Creating a Flag 
# --------------------------------------------------------------
# Define the earthquake period (adjust dates as needed)
earthquake_start = pd.to_datetime("2016-04-16")
earthquake_end = pd.to_datetime("2016-06-15")  # example period affected by earthquake relief

# Create an earthquake flag in the main dataframe
df['earthquake_flag'] = df['date'].apply(
    lambda d: 1 if earthquake_start <= d <= earthquake_end else 0
)

# --------------------------------------------------------------
# Preprocessing for time series
# --------------------------------------------------------------
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.day_name()  # Day of week
df["week"] = df["date"].dt.isocalendar().week  # Week number

# --------------------------------------------------------------
# 3) (Optional) Load Extra Datasets
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

try:
    df_stores = load_stores()
except FileNotFoundError:
    df_stores = None

try:
    df_oil = load_oil()
except FileNotFoundError:
    df_oil = None

try:
    df_holidays = load_holidays()
except FileNotFoundError:
    df_holidays = None

# --------------------------------------------------------------
# 4) Sidebar Filters for Core Data
# --------------------------------------------------------------
st.sidebar.header("Filter Options")

# Get the full list of stores and families
all_stores = sorted(df["store_nbr"].unique())
all_families = sorted(df["family"].unique())

# Initialize session state for widget keys if not already set
if "stores_select" not in st.session_state:
    st.session_state.stores_select = all_stores[:5]
if "families_select" not in st.session_state:
    st.session_state.families_select = ["GROCERY I"]

# Callback functions for select all buttons
def select_all_stores_callback():
    st.session_state.stores_select = all_stores

def select_all_families_callback():
    st.session_state.families_select = all_families

st.sidebar.button("Select All Stores", on_click=select_all_stores_callback)
st.sidebar.button("Select All Product Families", on_click=select_all_families_callback)

# Multiselect widgets using session_state as default values
selected_stores = st.sidebar.multiselect(
    "Select Store(s)",
    options=all_stores,
    default=st.session_state.stores_select,
    key="stores_select"
)
selected_families = st.sidebar.multiselect(
    "Select Product Family(ies)",
    options=all_families,
    default=st.session_state.families_select,
    key="families_select"
)

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
# 5) Create Daily Aggregation (used in EDA & decomposition)
# --------------------------------------------------------------
daily_sales = df_filtered.groupby("date")["sales"].sum().reset_index().sort_values("date")

# --------------------------------------------------------------
# 6) Tabs for Analysis
# --------------------------------------------------------------
tab_overview, tab_eda, tab_trends, tab_external, tab_protips, tab_forecast = st.tabs(
    ["Overview", "Advanced EDA", "Trends & Seasonality", "External Factors", "Pro Tips", "Forecast"]
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
    # Daily sales line chart (shows full historical data including earthquake period)
    st.subheader("Daily Total Sales")
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
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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
    # (D) Advanced Time Series Decomposition
    st.markdown("Decomposing daily sales into trend, seasonal, and residual components (using a weekly period).")
    daily_ts = daily_sales.set_index("date").asfreq("D").fillna(0)
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

# --------------------------------------------------------------
# Tab 4: External Factors
# --------------------------------------------------------------
with tab_external:
    st.subheader("External Factors Analysis")
    # 1) Oil CSV
    if df_oil is not None:
        st.markdown("### Oil Price Analysis")
        df_oil_renamed = df_oil.rename(columns={"dcoilwtico": "oil_price"})
        df_oil_merged = pd.merge(daily_sales, df_oil_renamed, on="date", how="left")
        fig_oil_line = px.line(df_oil_merged, x="date", y="oil_price", title="Daily Oil Price Over Time")
        st.plotly_chart(fig_oil_line, use_container_width=True)
        fig_oil_scatter = px.scatter(df_oil_merged, x="oil_price", y="sales", title="Oil Price vs. Sales", trendline="ols")
        st.plotly_chart(fig_oil_scatter, use_container_width=True)
    else:
        st.info("Oil CSV not found or not loaded.")
    # 2) Holidays
    if df_holidays is not None:
        st.markdown("### Holiday Effects on Sales")
        df_holidays_clean = df_holidays.copy()
        df_holidays_clean["is_holiday"] = 1
        df_hol_merged = pd.merge(daily_sales, df_holidays_clean[["date", "is_holiday"]], on="date", how="left")
        df_hol_merged["is_holiday"] = df_hol_merged["is_holiday"].fillna(0)
        df_hol_merged["HolidayFlag"] = df_hol_merged["is_holiday"].map({1: "Holiday", 0: "Non-Holiday"})
        fig_hol_box = px.box(df_hol_merged, x="HolidayFlag", y="sales", title="Sales on Holiday vs. Non-Holiday Days")
        st.plotly_chart(fig_hol_box, use_container_width=True)
    else:
        st.info("Holidays CSV not found or not loaded.")
    # 3) Stores metadata
    if df_stores is not None:
        st.markdown("### Store Clusters or Types")
        df_stores_merged = pd.merge(df_filtered, df_stores, on="store_nbr", how="left")
        cluster_group = df_stores_merged.groupby("cluster")["sales"].mean().reset_index()
        fig_cluster = px.bar(cluster_group, x="cluster", y="sales", title="Average Sales by Store Cluster", labels={"sales": "Avg Sales"})
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Stores CSV not found or not loaded.")
    # 4) Transactions
    try:
        df_transactions = pd.read_csv("transactions.csv", parse_dates=["date"])
    except FileNotFoundError:
        df_transactions = None
    if df_transactions is not None:
        st.markdown("### Transactions vs. Sales")
        df_trans_merged = pd.merge(df_filtered, df_transactions, on=["date", "store_nbr"], how="left")
        fig_trans_scatter = px.scatter(df_trans_merged, x="transactions", y="sales", title="Transactions vs. Sales", opacity=0.5)
        st.plotly_chart(fig_trans_scatter, use_container_width=True)
        daily_trans = df_trans_merged.groupby("date")["transactions"].sum().reset_index()
        daily_merged = pd.merge(daily_sales, daily_trans, on="date", how="left")
        fig_trans_line = px.line(daily_merged, x="date", y=["sales", "transactions"], title="Daily Sales & Transactions Over Time")
        st.plotly_chart(fig_trans_line, use_container_width=True)
    else:
        st.info("Transactions CSV not found or not loaded.")

# --------------------------------------------------------------
# Tab 6: Forecast 
# --------------------------------------------------------------
with tab_forecast:
    st.subheader("Forecast fututure sales figures")
    
    st.markdown("""
    This forecast uses a hybrid model based on Fourier features, lag features, and a trend captured via a time index.
    The process is:
    1. Engineer Fourier, lag, and time index features.
    3. Train Linear Regression and XGBoost models.
    4. Iteratively forecast future days using a hybrid weighted average.
    """)

    # ---------------------------
    # Utility Functions for Feature Engineering
    # ---------------------------
    def add_fourier_features(df_local, K=6, period=365):
        """Add K pairs of Fourier terms for yearly seasonality."""
        df_local['day_of_year'] = df_local['date'].dt.dayofyear
        for k in range(1, K+1):
            df_local[f'fourier_sin_k{k}'] = np.sin(2 * np.pi * k * df_local['day_of_year'] / period)
            df_local[f'fourier_cos_k{k}'] = np.cos(2 * np.pi * k * df_local['day_of_year'] / period)
        df_local.drop('day_of_year', axis=1, inplace=True)
        return df_local

    def add_lag_features_single(df_local, lags, target_col='sales'):
        """Add lag features for a single aggregated time series (no grouping).
           Expects df_local sorted by date."""
        for lag in lags:
            df_local[f'{target_col}_lag{lag}'] = df_local[target_col].shift(lag)
        return df_local

    # ---------------------------
    # 1) Parameters from UI
    # ---------------------------
    future_days = st.number_input("Days to Forecast into Future", min_value=1, max_value=1825, value=14)

    # Set number of Fourier pairs and lags.
    K_val = 6
    LAGS = [1, 7, 14, 21, 28]

    # ---------------------------
    # 2) Prepare Aggregated Data for Training the Model (Exclude Earthquake Period)
    # ---------------------------
    daily_sales_clean = daily_sales[~daily_sales['date'].between(earthquake_start, earthquake_end)]
    df_model = daily_sales_clean.copy()  # columns: date, sales
    df_model = df_model.sort_values("date")

    # Add time_index to capture overall trend 
    df_model["time_index"] = (df_model["date"] - df_model["date"].min()).dt.days

    # Process holidays from df_holidays (if available)
    if df_holidays is not None:
        df_hol = df_holidays.copy()
        df_hol['transferred'] = df_hol['transferred'].fillna('').astype(str)
        # Flag as holiday if not a transfer day
        df_hol['is_holiday'] = np.where(df_hol['transferred'].str.lower() == 'transfer', 0, 1)
        df_hol_valid = df_hol[df_hol['is_holiday'] == 1]
        # Create a set of month-day strings for holidays (e.g., "12-25")
        holiday_md = set(df_hol_valid['date'].dt.strftime('%m-%d'))
    else:
        st.info("Holidays CSV not loaded; holiday adjustments will be skipped.")
        holiday_md = set()

    # Add holiday_indicator to training data: 1 if date matches a known holiday, else 0
    df_model['holiday_indicator'] = df_model['date'].apply(lambda d: 1 if d.strftime('%m-%d') in holiday_md else 0)

    # ---------------------------
    # 3) Feature Engineering: Fourier, Lag, and Trend Features
    # ---------------------------
    df_model = add_fourier_features(df_model, K=K_val)
    df_model = add_lag_features_single(df_model, LAGS, target_col='sales')
    df_model.dropna(inplace=True)

    # Define features: lag features, holiday indicator, Fourier terms, and time_index.
    lag_features = [f'sales_lag{l}' for l in LAGS]
    fourier_features = [f'fourier_sin_k{k}' for k in range(1, K_val+1)] + [f'fourier_cos_k{k}' for k in range(1, K_val+1)]
    features = lag_features + ['holiday_indicator'] + fourier_features + ['time_index']

    X_train = df_model[features]
    y_train = df_model['sales']

    # ---------------------------
    # 4) Train Models 
    # ---------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    xgb_model = XGBRegressor(n_estimators=200, max_depth=50, learning_rate=0.4,
                             subsample=0.8, min_child_weight=0.5, random_state=42)
    xgb_model.fit(X_train, y_train)

    # ---------------------------
    # 5) Calculate the 365-Day Moving Average (for trend extraction)
    # ---------------------------
    df_model['moving_avg_365'] = df_model['sales'].rolling(window=365, min_periods=1).mean()

    #  Fit a linear trend to the historical 365-day moving average 
    time_indices = df_model["time_index"].values
    moving_avg = df_model["moving_avg_365"].values
    slope, intercept = np.polyfit(time_indices, moving_avg, 1)
    # The baseline is the last historical moving average value.
    baseline_ma = df_model["moving_avg_365"].iloc[-1]

    # ---------------------------
    # 6) Forecast Next X Days (Iterative Approach)
    # ---------------------------
    last_train_date = df_model['date'].max()
    df_future = df_model.copy()
    forecast_rows = []

    for step in range(1, future_days + 1):
        next_date = last_train_date + pd.Timedelta(days=step)
        new_row = {'date': next_date}
        # Fourier features for next_date
        day_of_year = next_date.timetuple().tm_yday
        for k in range(1, K_val+1):
            new_row[f'fourier_sin_k{k}'] = np.sin(2 * np.pi * k * day_of_year / 365)
            new_row[f'fourier_cos_k{k}'] = np.cos(2 * np.pi * k * day_of_year / 365)
        # Holiday indicator for next_date: based on month-day
        new_row['holiday_indicator'] = 1 if next_date.strftime('%m-%d') in holiday_md else 0
        # Set earthquake_flag to 0 for forecasting (assuming no future earthquake anomaly)
        new_row['earthquake_flag'] = 0
        # Add time_index for next_date
        new_row['time_index'] = (next_date - df_model['date'].min()).days

        # Compute lag features from the most recent available data in df_future
        for lag in LAGS:
            prev_date = next_date - pd.Timedelta(days=lag)
            prev_sales_row = df_future.loc[df_future['date'] == prev_date]
            if len(prev_sales_row) == 1:
                new_row[f'sales_lag{lag}'] = prev_sales_row['sales'].values[0]
            else:
                new_row[f'sales_lag{lag}'] = 0.0

        # Build feature vector for prediction
        df_new = pd.DataFrame([new_row])
        X_new = df_new[features]

        # Model Forecast 
        lr_pred  = lr.predict(X_new)[0]
        xgb_pred = xgb_model.predict(X_new)[0]
        hybrid_pred = 0.12 * lr_pred + 0.88 * xgb_pred

        # Clip negative values
        pred_clipped = max(hybrid_pred, 0)

        
        moving_avg_last = df_model['moving_avg_365'].iloc[-1]
        pred_with_trend = pred_clipped * (moving_avg_last / max(moving_avg_last, 1e-5))

        # Post-Process Forecast with Linear Trend 
        forecast_time_index = new_row['time_index']
        predicted_ma = intercept + slope * forecast_time_index
        amp_factor = 0.2  # Experiment with values > 1 to boost amplitude
        final_pred = pred_with_trend + amp_factor * (predicted_ma - baseline_ma)

        # Special adjustment for New Year's Day.
        if next_date.month == 1 and next_date.day == 1:
            final_pred = final_pred * 0.0164

        # Easter Sunday Adjustment 
        easter_date = easter.easter(next_date.year)
        if next_date.date() == easter_date:
            final_pred = final_pred * (1 - 0.0556)

        df_new['sales'] = final_pred
        forecast_rows.append((next_date, final_pred))
        df_future = pd.concat([df_future, df_new], ignore_index=True)

    # ---------------------------
    # 7) Combine Historical & Future Data for Plotting
    # ---------------------------
    forecast_df = pd.DataFrame(forecast_rows, columns=['date', 'forecast'])

    fig_future_fc = go.Figure()
    fig_future_fc.add_trace(go.Scatter(
        x=daily_sales['date'],
        y=daily_sales['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='black', width=2)
    ))
    fig_future_fc.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecasted Sales',
        line=dict(color='blue', dash='dash')
    ))
    fig_future_fc.update_layout(
        title=f"Forecast for the Next {future_days} Days",
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(fig_future_fc, use_container_width=True)

    st.markdown(f"""
    The hybrid model was trained on **{len(df_model)}** rows of aggregated historical data
    (with Fourier, lag, holiday, and trend features; earthquake data was excluded for training).
    Forecasted **{future_days}** days beyond {last_train_date.date()}.
    """)
    st.markdown("#### Forecasted Values")
    st.write(forecast_df)

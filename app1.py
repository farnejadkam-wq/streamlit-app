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
import os

# --------------------------------------------------------------
# 1) Streamlit Page Config
# --------------------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Explorer",
    layout="wide"
)

st.title("Store Sales - Data Exploration & Time Series Tool")
st.caption("**Dataset:** Kaggle's Store Sales - Time Series Forecasting")

# --------------------------------------------------------------
# 2) Data Loading with Optimizations
# --------------------------------------------------------------
# Create processed_data directory if it doesn't exist
os.makedirs("processed_data", exist_ok=True)

@st.cache_data(ttl=3600, show_spinner=False)
def load_train_data(sample_fraction=None):
    """Load training data with optional sampling for faster processing"""
    try:
        # First try to load preprocessed data if available
        if os.path.exists("processed_data/train_processed.csv"):
            df_train = pd.read_csv("processed_data/train_processed.csv", parse_dates=["date"])
            st.success("Using preprocessed data")
            return df_train
    except Exception as e:
        st.warning(f"Could not load preprocessed data: {e}")
    
    # Fall back to original file with sampling
    try:
        df_train = pd.read_csv("train.csv", parse_dates=["date"])
        if sample_fraction and sample_fraction < 1.0:
            df_train = df_train.sample(frac=sample_fraction, random_state=42)
            st.info(f"Loaded {sample_fraction*100:.0f}% sample of data for better performance")
        return df_train
    except FileNotFoundError:
        st.error("Could not find `train.csv` in the current directory. Please place it next to `app.py`.")
        st.stop()

# Load data with sampling option - adjust based on your needs
with st.spinner("Loading data..."):
    # Use a smaller sample for better performance on cloud hosting
    df = load_train_data(sample_fraction=0.25)  # Using 25% of the data

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
# Preprocessing for time series - more efficient approach
# --------------------------------------------------------------
# Add features using vectorized operations instead of apply()
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.day_name()  # Day of week
df["week"] = df["date"].dt.isocalendar().week  # Week number

# --------------------------------------------------------------
# 3) Load Extra Datasets with Better Caching and Error Handling
# --------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_stores():
    try:
        return pd.read_csv("stores.csv")
    except FileNotFoundError:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_oil():
    try:
        return pd.read_csv("oil.csv", parse_dates=["date"])
    except FileNotFoundError:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_holidays():
    try:
        return pd.read_csv("holidays_events.csv", parse_dates=["date"])
    except FileNotFoundError:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_transactions():
    try:
        return pd.read_csv("transactions.csv", parse_dates=["date"])
    except FileNotFoundError:
        return None

# Load datasets with progress indication
with st.spinner("Loading additional datasets..."):
    df_stores = load_stores()
    df_oil = load_oil()
    df_holidays = load_holidays()
    df_transactions = load_transactions()

# --------------------------------------------------------------
# 4) Sidebar Filters with Session State Optimization
# --------------------------------------------------------------
st.sidebar.header("Filter Options")

# Get the full list of stores and families
# Reduce memory by sorting only what's needed
all_stores = sorted(df["store_nbr"].unique())
all_families = sorted(df["family"].unique())

# Make default selections smaller to reduce initial load
if "stores_select" not in st.session_state:
    st.session_state.stores_select = all_stores[:3]  # Default to just 3 stores
if "families_select" not in st.session_state:
    st.session_state.families_select = ["GROCERY I"]  # Default to just 1 family

# Callback functions for select all buttons
def select_all_stores_callback():
    st.session_state.stores_select = all_stores

def select_all_families_callback():
    st.session_state.families_select = all_families

# Add warning about performance with select all
if st.sidebar.button("Select All Stores", on_click=select_all_stores_callback):
    st.sidebar.warning("Selecting all stores may impact performance")

if st.sidebar.button("Select All Product Families", on_click=select_all_families_callback):
    st.sidebar.warning("Selecting all families may impact performance")

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

# Optimize date range selection by starting with a smaller default range
min_date, max_date = df["date"].min(), df["date"].max()
default_end = max_date
default_start = default_end - pd.Timedelta(days=90)  # Default to last 90 days for better performance
start_date, end_date = st.sidebar.date_input(
    "Select Date Range", 
    [max(min_date, default_start), default_end],
    min_value=min_date, 
    max_value=max_date
)

# Show warning if date range is too large
date_range_days = (end_date - start_date).days
if date_range_days > 365:
    st.sidebar.warning(f"Large date range selected ({date_range_days} days). This may slow down the app.")

# Apply filters more efficiently
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
# 5) Create Daily Aggregation with Caching
# --------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_daily_sales(df_filtered):
    return df_filtered.groupby("date")["sales"].sum().reset_index().sort_values("date")

daily_sales = get_daily_sales(df_filtered)

# --------------------------------------------------------------
# 6) Tabs for Analysis with Lazy Loading
# --------------------------------------------------------------
tab_overview, tab_eda, tab_trends, tab_external, tab_forecast = st.tabs(
    ["Overview", "Advanced EDA", "Trends & Seasonality", "External Factors", "Forecast"]
)

# --------------------------------------------------------------
# Tab 1: Overview - Always show this
# --------------------------------------------------------------
with tab_overview:
    st.subheader("Overview of Filtered Data")
    with st.expander("Preview Data Sample"):
        # Limit sample size for performance
        max_sample = min(20, len(df_filtered))
        st.write(df_filtered.head(max_sample))
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
    # Limit sample size for better performance
    sample_size = min(5000, len(df_filtered))
    scatter_df = df_filtered.sample(sample_size, random_state=42)
    fig_scatter = px.scatter(scatter_df, x="onpromotion", y="sales", opacity=0.6,
                             title="onpromotion vs. Sales (Sample)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------------------
# Tab 2: Advanced EDA - Lazy load with caching
# --------------------------------------------------------------
with tab_eda:
    st.subheader("Advanced Exploratory Data Analysis")
    
    @st.cache_data(ttl=3600)
    def get_store_family_avg(df_filtered):
        # More efficient groupby with optimized sample
        df_sample = df_filtered.sample(min(100000, len(df_filtered)), random_state=42)
        return df_sample.groupby(["store_nbr", "family"])["sales"].mean().reset_index()
    
    @st.cache_data(ttl=3600)
    def get_rolling_avg(df_filtered):
        daily_agg = df_filtered.groupby("date")["sales"].sum().reset_index().sort_values("date")
        daily_agg["rolling_7"] = daily_agg["sales"].rolling(window=7, min_periods=1).mean()
        return daily_agg
    
    # Only calculate when this tab is selected - load lazily
    if st.checkbox("Show Store & Family Analysis (Heavy Computation)", value=False):
        with st.spinner("Calculating averages..."):
            avg_by_sf = get_store_family_avg(df_filtered)
            fig_sf = px.bar(avg_by_sf, x="store_nbr", y="sales", color="family", barmode="group",
                            title="Average Sales by Store & Family")
            st.plotly_chart(fig_sf, use_container_width=True)
    
    # Rolling average - less heavy computation
    st.markdown("#### Rolling Weekly Average Sales")
    with st.spinner("Calculating rolling average..."):
        daily_agg = get_rolling_avg(df_filtered)
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["sales"],
                                      mode="lines", name="Daily Sales"))
        fig_roll.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["rolling_7"],
                                      mode="lines", name="7-day Rolling Avg"))
        fig_roll.update_layout(title="Daily Sales with 7-day Rolling Average",
                               xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig_roll, use_container_width=True)
    
    # Distribution of Sales - lighter computation
    st.markdown("#### Distribution of Sales")
    # Sample data for histogram for better performance
    hist_sample = df_filtered.sample(min(50000, len(df_filtered)), random_state=42)
    fig_hist = px.histogram(hist_sample, x="sales", nbins=50, title="Sales Distribution (Sample)")
    st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------------------------------------------
# Tab 3: Trends & Seasonality - Cached functions
# --------------------------------------------------------------
with tab_trends:
    st.subheader("Trends & Seasonality Analysis")
    
    @st.cache_data(ttl=3600)
    def get_dow_patterns(df_filtered):
        dow_group = df_filtered.groupby("dow")["sales"].mean().reset_index()
        ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_group["dow"] = pd.Categorical(dow_group["dow"], categories=ordered_days, ordered=True)
        dow_group.sort_values("dow", inplace=True)
        return dow_group
    
    @st.cache_data(ttl=3600)
    def get_monthly_patterns(df_filtered):
        return df_filtered.groupby("month")["sales"].mean().reset_index()
    
    @st.cache_data(ttl=3600)
    def get_yearly_trends(df_filtered):
        return df_filtered.groupby("year")["sales"].mean().reset_index()
    
    # Day-of-Week Patterns
    st.markdown("#### Average Sales by Day of Week")
    with st.spinner("Analyzing day of week patterns..."):
        dow_group = get_dow_patterns(df_filtered)
        fig_dow = px.bar(dow_group, x="dow", y="sales", title="Avg Sales by Day of Week", color="dow")
        st.plotly_chart(fig_dow, use_container_width=True)
    
    # Monthly Patterns
    st.markdown("#### Monthly Patterns")
    with st.spinner("Analyzing monthly patterns..."):
        mon_group = get_monthly_patterns(df_filtered)
        fig_mon = px.line(mon_group, x="month", y="sales", title="Avg Sales by Month", markers=True)
        fig_mon.update_xaxes(dtick=1)
        st.plotly_chart(fig_mon, use_container_width=True)
    
    # Yearly Trends
    st.markdown("#### Yearly Trends")
    with st.spinner("Analyzing yearly trends..."):
        year_group = get_yearly_trends(df_filtered)
        fig_year = px.bar(year_group, x="year", y="sales", title="Avg Sales by Year", color="year")
        st.plotly_chart(fig_year, use_container_width=True)
    
    # Time Series Decomposition - only if requested (heavy computation)
    if st.checkbox("Run Time Series Decomposition (Heavy Computation)", value=False):
        st.markdown("Decomposing daily sales into trend, seasonal, and residual components (using a weekly period).")
        with st.spinner("Performing time series decomposition..."):
            # Use a reduced dataset for decomposition
            max_days = 365  # Limit to at most 1 year
            daily_ts = daily_sales.tail(max_days).set_index("date").asfreq("D").fillna(0)
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
# Tab 4: External Factors - Lazy load with user action
# --------------------------------------------------------------
with tab_external:
    st.subheader("External Factors Analysis")
    
    if st.checkbox("Analyze External Factors (Oil, Holidays, etc.)", value=False):
        # 1) Oil CSV
        if df_oil is not None:
            st.markdown("### Oil Price Analysis")
            with st.spinner("Analyzing oil price relationship..."):
                df_oil_renamed = df_oil.rename(columns={"dcoilwtico": "oil_price"})
                # Filter oil data to match the date range for better performance
                df_oil_filtered = df_oil_renamed[
                    (df_oil_renamed["date"] >= pd.to_datetime(start_date)) &
                    (df_oil_renamed["date"] <= pd.to_datetime(end_date))
                ]
                df_oil_merged = pd.merge(daily_sales, df_oil_filtered, on="date", how="left")
                fig_oil_line = px.line(df_oil_merged, x="date", y="oil_price", title="Daily Oil Price Over Time")
                st.plotly_chart(fig_oil_line, use_container_width=True)
                
                # Sample for scatter plot
                df_oil_sample = df_oil_merged.sample(min(5000, len(df_oil_merged)), random_state=42)
                fig_oil_scatter = px.scatter(df_oil_sample, x="oil_price", y="sales", 
                                           title="Oil Price vs. Sales (Sample)", trendline="ols")
                st.plotly_chart(fig_oil_scatter, use_container_width=True)
        else:
            st.info("Oil CSV not found or not loaded.")
        
        # 2) Holidays - only show if data available
        if df_holidays is not None:
            st.markdown("### Holiday Effects on Sales")
            with st.spinner("Analyzing holiday effects..."):
                df_holidays_clean = df_holidays.copy()
                df_holidays_clean["is_holiday"] = 1
                # Filter holidays to the date range for better performance
                df_holidays_filtered = df_holidays_clean[
                    (df_holidays_clean["date"] >= pd.to_datetime(start_date)) &
                    (df_holidays_clean["date"] <= pd.to_datetime(end_date))
                ]
                df_hol_merged = pd.merge(daily_sales, df_holidays_filtered[["date", "is_holiday"]], on="date", how="left")
                df_hol_merged["is_holiday"] = df_hol_merged["is_holiday"].fillna(0)
                df_hol_merged["HolidayFlag"] = df_hol_merged["is_holiday"].map({1: "Holiday", 0: "Non-Holiday"})
                fig_hol_box = px.box(df_hol_merged, x="HolidayFlag", y="sales", title="Sales on Holiday vs. Non-Holiday Days")
                st.plotly_chart(fig_hol_box, use_container_width=True)
        else:
            st.info("Holidays CSV not found or not loaded.")
        
        # 3) Stores metadata - only show if data available
        if df_stores is not None:
            st.markdown("### Store Clusters or Types")
            with st.spinner("Analyzing store clusters..."):
                df_stores_merged = pd.merge(df_filtered, df_stores, on="store_nbr", how="left")
                # Use sample for better performance
                df_stores_sample = df_stores_merged.sample(min(50000, len(df_stores_merged)), random_state=42)
                cluster_group = df_stores_sample.groupby("cluster")["sales"].mean().reset_index()
                fig_cluster = px.bar(cluster_group, x="cluster", y="sales", title="Average Sales by Store Cluster", 
                                  labels={"sales": "Avg Sales"})
                st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info("Stores CSV not found or not loaded.")
        
        # 4) Transactions - only show if data available
        if df_transactions is not None:
            st.markdown("### Transactions vs. Sales")
            with st.spinner("Analyzing transactions relationship..."):
                # Filter transactions to match the date range for better performance
                df_trans_filtered = df_transactions[
                    (df_transactions["date"] >= pd.to_datetime(start_date)) &
                    (df_transactions["date"] <= pd.to_datetime(end_date)) &
                    (df_transactions["store_nbr"].isin(selected_stores))
                ]
                df_trans_merged = pd.merge(df_filtered, df_trans_filtered, on=["date", "store_nbr"], how="left")
                
                # Sample for scatter plot
                trans_sample = df_trans_merged.sample(min(5000, len(df_trans_merged)), random_state=42)
                fig_trans_scatter = px.scatter(trans_sample, x="transactions", y="sales", 
                                             title="Transactions vs. Sales (Sample)", opacity=0.5)
                st.plotly_chart(fig_trans_scatter, use_container_width=True)
                
                daily_trans = df_trans_filtered.groupby("date")["transactions"].sum().reset_index()
                daily_merged = pd.merge(daily_sales, daily_trans, on="date", how="left")
                fig_trans_line = px.line(daily_merged, x="date", y=["sales", "transactions"], 
                                       title="Daily Sales & Transactions Over Time")
                st.plotly_chart(fig_trans_line, use_container_width=True)
        else:
            st.info("Transactions CSV not found or not loaded.")
    else:
        st.info("Check the box above to analyze external factors")

# --------------------------------------------------------------
# Tab 5: Forecast - Only run when button is clicked
# --------------------------------------------------------------
with tab_forecast:
    st.subheader("Forecast future sales figures")
    
    st.markdown("""
    This forecast uses a hybrid model based on Fourier features, lag features, and a trend captured via a time index.
    Click the button below to run the forecast.
    """)

    # Only run forecast if button is clicked
    if st.button("Generate Forecast"):
        # Limit forecast days for better performance
        future_days = st.number_input("Days to Forecast into Future", min_value=1, max_value=30, value=14)
        
        with st.spinner("Building forecast model - this may take a minute..."):
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
            # Use fewer Fourier terms and lags for performance
            # ---------------------------
            K_val = 3  # Reduced from 6
            LAGS = [1, 7, 14]  # Reduced from [1, 7, 14, 21, 28]

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
            # 4) Train Models with simplified parameters
            # ---------------------------
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            # Use reduced parameters for XGBoost to improve performance
            xgb_model = XGBRegressor(
                n_estimators=100,  # Reduced from 200
                max_depth=8,       # Reduced from 50
                learning_rate=0.1,  # Reduced from 0.4
                subsample=0.8, 
                min_child_weight=0.5, 
                random_state=42
            )
            xgb_model.fit(X_train, y_train)

            # ---------------------------
            # 5) Calculate the 365-Day Moving Average (for trend extraction)
            # ---------------------------
            window_size = min(365, len(df_model))  # Use smaller window if data is limited
            df_model['moving_avg_365'] = df_model['sales'].rolling(window=window_size, min_periods=1).mean()

            #  Fit a linear trend to the historical moving average 
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
                lr_pred = lr.predict(X_new)[0]
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
            
            # Add download button for the forecast
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name="sales_forecast.csv",
                mime="text/csv",
            )
    else:
        st.info("Click 'Generate Forecast' button to run the forecast model (this may take a minute to process)")

# --------------------------------------------------------------
# Add Data Preprocessing Tool
# --------------------------------------------------------------
if st.sidebar.checkbox("Show Data Preprocessing Tools", value=False):
    st.sidebar.subheader("Data Preprocessing")
    st.sidebar.info("These tools help create optimized versions of your data for better app performance.")
    
    if st.sidebar.button("Preprocess Data"):
        with st.sidebar:
            with st.spinner("Running preprocessing..."):
                # Create directory for processed data
                os.makedirs("processed_data", exist_ok=True)
                
                # 1. Create a reduced version of the main dataset
                try:
                    full_df = pd.read_csv("train.csv", parse_dates=["date"])
                    # Create a 25% sample
                    sample_df = full_df.sample(frac=0.25, random_state=42)
                    sample_df.to_csv("processed_data/train_processed.csv", index=False)
                    st.sidebar.success(f"Created train_processed.csv with {len(sample_df):,} rows")
                    
                    # 2. Create pre-aggregated data
                    # Daily aggregation
                    daily_agg = full_df.groupby("date")["sales"].sum().reset_index()
                    daily_agg.to_csv("processed_data/daily_sales_agg.csv", index=False)
                    st.sidebar.success(f"Created daily_sales_agg.csv")
                    
                    # Store-level aggregation
                    store_agg = full_df.groupby(["store_nbr", "date"])["sales"].sum().reset_index()
                    store_agg.to_csv("processed_data/store_sales_agg.csv", index=False)
                    st.sidebar.success(f"Created store_sales_agg.csv")
                    
                    # Family-level aggregation
                    family_agg = full_df.groupby(["family", "date"])["sales"].sum().reset_index()
                    family_agg.to_csv("processed_data/family_sales_agg.csv", index=False)
                    st.sidebar.success(f"Created family_sales_agg.csv")
                    
                    st.sidebar.success("Preprocessing complete! Restart the app to use the optimized data.")
                except Exception as e:
                    st.sidebar.error(f"Error during preprocessing: {e}")

# --------------------------------------------------------------
# Show App Information
# --------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("""
### App Performance Tips
1. Select fewer stores and product families
2. Use a shorter date range
3. Run preprocessing once to create optimized data files
4. For heavy computations, use sample data
""")

# --------------------------------------------------------------
# Footer with Memory Usage Info
# --------------------------------------------------------------
st.sidebar.markdown("---")
import psutil
process = psutil.Process()
memory_info = process.memory_info()
memory_mb = memory_info.rss / 1024 / 1024
st.sidebar.caption(f"Memory Usage: {memory_mb:.1f} MB")

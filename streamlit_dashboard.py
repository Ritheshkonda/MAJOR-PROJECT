# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

LOG_FILE = "emotion_log.csv"

st.set_page_config(page_title="Emotion Dashboard", layout="wide")
st.title("Emotion Dashboard — Real-time Overview")

@st.cache_data
def load_data(path=LOG_FILE):
    """Load and sanitize the CSV. Always return a DataFrame with expected columns."""
    cols = ["timestamp", "face_id", "label", "confidence", "x", "y", "w", "h"]
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        # safe conversion; if parse failed earlier, force parse here
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # ensure required columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        # drop rows missing timestamp or label (can't process them)
        df = df.dropna(subset=["timestamp", "label"]).reset_index(drop=True)
        # normalize label to str
        df["label"] = df["label"].astype(str)
        return df[cols]
    except FileNotFoundError:
        return pd.DataFrame(columns=cols)
    except Exception:
        # return empty frame with correct columns on any error
        return pd.DataFrame(columns=cols)

# manual reload button
if st.sidebar.button("Reload data"):
    load_data.clear()  # clear cache so load_data reads file again

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
labels = sorted(df['label'].unique()) if not df.empty else []
selected_labels = st.sidebar.multiselect("Show labels", options=labels, default=labels)

if df.empty:
    st.warning("No log data found yet. Run realtime logger first or check emotion_log.csv.")
    st.stop()

# apply label filter (if user cleared selection, show none)
if selected_labels:
    df = df[df['label'].isin(selected_labels)]
else:
    df = df.iloc[0:0]  # empty if nothing selected

# Summary cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Unique Faces Observed", int(df['face_id'].nunique() if "face_id" in df.columns else 0))
last_update = df['timestamp'].max() if not df.empty else pd.NaT
col3.metric("Last Update", last_update.strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(last_update) else "No data")

# Pie chart of label distribution
if not df.empty:
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    fig_pie = px.pie(label_counts, values='count', names='label', title="Emotion Distribution")
    st.plotly_chart(fig_pie, use_container_width=False, width='stretch')
else:
    st.info("No labels to show in distribution (check filters).")

# Time series: per-minute counts
# Robust grouping: use Grouper + groupby to get counts per minute per label
if not df.empty:
    # ensure timestamp index and timezone-aware dtype are handled
    df_ts = (
        df
        .set_index('timestamp')
        .groupby([pd.Grouper(freq='1Min'), 'label'])
        .size()
        .unstack(fill_value=0)
    )
    if not df_ts.empty:
        st.subheader("Emotion counts over time (per minute)")
        # reset_index to get 'timestamp' column for plotting
        ts_plot_df = df_ts.reset_index()
        # px.line accepts a list of columns as y
        y_cols = df_ts.columns.tolist()
        fig_ts = px.line(ts_plot_df, x='timestamp', y=y_cols, labels={'value': 'count', 'timestamp': 'time'})
        st.plotly_chart(fig_ts, use_container_width=False, width='stretch')
    else:
        st.info("No per-minute data to plot.")
else:
    st.info("No time-series data available.")

# Recent detections table
st.subheader("Recent detections")
if not df.empty:
    st.dataframe(df.sort_values('timestamp', ascending=False).head(200))
else:
    st.write("No recent detections to display.")

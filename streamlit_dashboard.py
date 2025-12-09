# streamlit_dashboard.py (fixed & robust)
import streamlit as st
import pandas as pd
import plotly.express as px
import os

LOG_FILE = "emotion_log.csv"

st.set_page_config(page_title="Emotion Dashboard", layout="wide")
st.title("Emotion Dashboard — Real-time Overview")

@st.cache_data
def load_data(path=LOG_FILE):
    cols_expected = ["timestamp", "face_id", "label", "confidence", "x", "y", "w", "h"]
    try:
        # read raw csv (no parse yet)
        df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return pd.DataFrame(columns=cols_expected)

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # accept either 'timestamp' or 'timestamp_utc'
    if "timestamp" not in df.columns and "timestamp_utc" in df.columns:
        df = df.rename(columns={"timestamp_utc": "timestamp"})
    if "timestamp" not in df.columns:
        # if no timestamp column, return empty frame with expected cols
        return pd.DataFrame(columns=cols_expected)

    # convert types and sanitize
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["label"] = df.get("label", "").astype(str)
    # coerce numeric columns if present
    for c in ["face_id", "confidence", "x", "y", "w", "h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    df = df.dropna(subset=["timestamp", "label"]).reset_index(drop=True)

    # ensure column order
    return df[cols_expected]

if st.sidebar.button("Reload data"):
    load_data.clear()

df = load_data()

st.sidebar.header("Filters")
labels = sorted(df['label'].unique()) if not df.empty else []
selected_labels = st.sidebar.multiselect("Show labels", options=labels, default=labels)

if df.empty:
    st.warning("No log data found yet. Run realtime logger first or check emotion_log.csv.")
    st.stop()

if selected_labels:
    df = df[df['label'].isin(selected_labels)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Unique Faces Observed", int(df['face_id'].nunique() if "face_id" in df.columns else 0))
last_update = df['timestamp'].max() if not df.empty else pd.NaT
col3.metric("Last Update", last_update.strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(last_update) else "No data")

if not df.empty:
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    fig_pie = px.pie(label_counts, values='count', names='label', title="Emotion Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("No labels to show in distribution (check filters).")

if not df.empty:
    df_ts = (
        df
        .set_index('timestamp')
        .groupby([pd.Grouper(freq='1Min'), 'label'])
        .size()
        .unstack(fill_value=0)
    )
    if not df_ts.empty:
        st.subheader("Emotion counts over time (per minute)")
        ts_plot_df = df_ts.reset_index()
        y_cols = df_ts.columns.tolist()
        fig_ts = px.line(ts_plot_df, x='timestamp', y=y_cols, labels={'value': 'count', 'timestamp': 'time'})
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No per-minute data to plot.")
else:
    st.info("No time-series data available.")

st.subheader("Recent detections")
st.dataframe(df.sort_values('timestamp', ascending=False).head(200))

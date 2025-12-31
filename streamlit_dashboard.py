# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from io import BytesIO

LOG_FILE = "emotion_log.csv"
REC_FILE = "recommendations.csv"

st.set_page_config(page_title="Emotion Dashboard", layout="wide")
st.title("Emotion Dashboard — Real-time Overview")

@st.cache_data
def load_emotion_log(path=LOG_FILE):
    expected_cols = ["timestamp", "face_id", "label", "confidence", "x", "y", "w", "h"]
    try:
        raw = pd.read_csv(path, dtype=str, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame(columns=expected_cols)
    raw.columns = [c.strip() for c in raw.columns]

    if "timestamp" not in raw.columns and "timestamp_utc" in raw.columns:
        raw = raw.rename(columns={"timestamp_utc": "timestamp"})
    if "timestamp" not in raw.columns:
        return pd.DataFrame(columns=expected_cols)

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw["label"] = raw.get("label", "").astype(str)

    for c in ["face_id", "confidence", "x", "y", "w", "h"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        else:
            raw[c] = pd.NA

    df = raw.dropna(subset=["timestamp", "label"]).reset_index(drop=True)
    return df[expected_cols]

@st.cache_data
def load_recommendations(path=REC_FILE):
    cols = ["timestamp", "face_id", "label", "score", "intent", "advice", "priority"]
    try:
        raw = pd.read_csv(path, dtype=str, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame(columns=cols)
    raw.columns = [c.strip() for c in raw.columns]
    if "timestamp" not in raw.columns and "timestamp_utc" in raw.columns:
        raw = raw.rename(columns={"timestamp_utc": "timestamp"})
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    for c in ["face_id", "score", "priority"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        else:
            raw[c] = pd.NA
    if "label" not in raw.columns:
        raw["label"] = ""
    return raw[cols]

if st.sidebar.button("Reload data"):
    load_emotion_log.clear()
    load_recommendations.clear()

df = load_emotion_log()
recs = load_recommendations()

st.sidebar.header("Filters")
labels = sorted(df['label'].unique()) if not df.empty else []
selected_labels = st.sidebar.multiselect("Show labels", options=labels, default=labels)

min_time = None
max_time = None
if not df.empty:
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    start_dt, end_dt = st.sidebar.date_input("Date range", [min_time.date(), max_time.date()]) if min_time and max_time else (None, None)
else:
    start_dt, end_dt = (None, None)

if df.empty:
    st.warning("No log data found yet. Run realtime logger first or check emotion_log.csv.")
else:
    if selected_labels:
        df = df[df['label'].isin(selected_labels)]

    if start_dt and end_dt:
        start_ts = pd.to_datetime(start_dt).tz_localize("UTC")
        end_ts = pd.to_datetime(end_dt).tz_localize("UTC") + pd.Timedelta(days=1)
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] < end_ts)]

# Top-line metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
col2.metric("Unique Faces Observed", int(df['face_id'].nunique() if not df.empty and 'face_id' in df.columns else 0))
last_update = df['timestamp'].max() if not df.empty else pd.NaT
col3.metric("Last Update (UTC)", last_update.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(last_update) else "No data")

# Engagement score: simple normalized metric
def compute_engagement(dataframe):
    if dataframe.empty:
        return None
    pos = dataframe['label'].isin(['happy', 'surprise']).sum()
    neg = dataframe['label'].isin(['fear', 'sad', 'angry', 'disgust']).sum()
    total = max(1, len(dataframe))
    score = (pos - neg) / total  # -1 .. 1
    # convert to 0..100
    return max(0, min(100, int((score + 1) * 50)))

eng_score = compute_engagement(df)
col4.metric("Engagement (0-100)", eng_score if eng_score is not None else "N/A")

st.markdown("---")

# Emotion distribution pie
if not df.empty:
    label_counts = df['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    fig_pie = px.pie(label_counts, values='count', names='label', title="Emotion Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("No labels to show in distribution.")

# Time series per-minute
if not df.empty:
    df_ts = df.set_index('timestamp').groupby([pd.Grouper(freq='1Min'), 'label']).size().unstack(fill_value=0)
    if not df_ts.empty:
        st.subheader("Emotion counts over time (per minute)")
        ts_plot_df = df_ts.reset_index()
        y_cols = df_ts.columns.tolist()
        fig_ts = go.Figure()
        for col in y_cols:
            fig_ts.add_trace(go.Scatter(x=ts_plot_df['timestamp'], y=ts_plot_df[col], mode='lines', name=col))
        fig_ts.update_layout(xaxis_title='Time (UTC)', yaxis_title='Count', legend_title='Emotion')
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No per-minute data to plot.")
else:
    st.info("No time-series data available.")

# Per-student summary
st.subheader("Per-Student Summary")
if not df.empty and 'face_id' in df.columns:
    per_student = df.groupby('face_id').agg(
        total_detections=('label', 'count'),
        most_common=('label', lambda x: x.mode().iat[0] if not x.mode().empty else ""),
        avg_confidence=('confidence', 'mean')
    ).reset_index()
    per_student['avg_confidence'] = per_student['avg_confidence'].round(3)
    per_student = per_student.sort_values('total_detections', ascending=False)
    st.dataframe(per_student.head(200))
    csv = per_student.to_csv(index=False).encode('utf-8')
    st.download_button("Download per-student summary CSV", data=csv, file_name="per_student_summary.csv", mime="text/csv")
else:
    st.info("No per-student data to display.")

st.markdown("---")

# Recent detections and ability to download full log slice
st.subheader("Recent Detections")
if not df.empty:
    recent = df.sort_values('timestamp', ascending=False).head(500)
    st.dataframe(recent)
    buf = recent.to_csv(index=False).encode('utf-8')
    st.download_button("Download recent detections (CSV)", data=buf, file_name="recent_detections.csv", mime="text/csv")
else:
    st.write("No recent detections to display.")

st.markdown("---")

# Recommendations panel
st.subheader("Recent Recommendations")
if recs.empty:
    st.info("No recommendations logged yet. Run the adaptive realtime logger first.")
else:
    # apply same date filter if possible
    if "timestamp" in recs.columns and not recs['timestamp'].isna().all():
        recs = recs.sort_values('timestamp', ascending=False)
    st.dataframe(recs.head(200))
    buf2 = recs.head(200).to_csv(index=False).encode('utf-8')
    st.download_button("Download recommendations CSV", data=buf2, file_name="recommendations_recent.csv", mime="text/csv")

st.markdown("---")
st.write("Tips: Use the realtime adaptive logger `realtime_multi_face_adaptive.py` to populate both logs. Use the Reload button in the sidebar after starting the logger.")

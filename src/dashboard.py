import streamlit as st
import sqlite3
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="Prompt Optimization Telemetry", layout="wide", page_icon="⚡")

def load_telemetry_data():
    # Force direct absolute mapping to break out of relative directory execution traps
    absolute_db_path = r"C:\Users\srina\Downloads\PROMPT OPTIMIZATION\project\data\results.db"
    
    # Fallback paths list just in case of environment shifts
    possible_paths = [
        absolute_db_path,
        "data/results.db",
        "../data/results.db",
        os.path.join(os.getcwd(), "data", "results.db")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                conn = sqlite3.connect(path)
                # Pull the raw data table directly
                df = pd.read_sql_query("SELECT * FROM benchmark_results", conn)
                conn.close()
                if not df.empty:
                    return df
            except Exception as e:
                continue
                
    # If the file isn't found across any paths, visually print the lookups for fast debugging
    if not os.path.exists(absolute_db_path):
        st.sidebar.error(f"⚠️ Target DB file not found at: {absolute_db_path}")
        
    return pd.DataFrame()

st.title("⚡ Carbon-Aware Prompt Optimization Engine Proxy Telemetry")
st.markdown("### Real-time operational engineering metrics & token healing analytics dashboard")

df = load_telemetry_data()

# 2. Check if data actually exists
if df.empty:
    st.info("🔄 Awaiting incoming proxy gateway traffic metrics. Run your `test_client` script to stream database telemetry logs.")
else:
    # 3. Dynamic KPI Strip Calculations
    total_requests = len(df)
    
    # Safely extract token savings using your exact column name
    total_tokens_saved = df["token_delta"].sum() if "token_delta" in df.columns else 0
    
    # Calculate fidelity percentages safely from bert_f1_score
    if "bert_f1_score" in df.columns:
        avg_fidelity = df["bert_f1_score"].mean() * 100
        # Handle cases where score might already be stored as a percentage
        if avg_fidelity > 100:
            avg_fidelity = df["bert_f1_score"].mean()
    else:
        avg_fidelity = 0.0

    total_carbon_saved = df["carbon_footprint_g"].sum() if "carbon_footprint_g" in df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Network API Requests", value=f"{total_requests}")
    with col2:
        st.metric(label="Total LLM Input Tokens Saved", value=f"{total_tokens_saved:,}")
    with col3:
        st.metric(label="Average Semantic Fidelity (BERTScore)", value=f"{avg_fidelity:.2f}%")
    with col4:
        st.metric(label="Total Est. Carbon Footprint Offset", value=f"{total_carbon_saved:.4f} g CO2e")

    st.markdown("---")

    # Hardware Efficiency Profile Metrics
    max_ram_spike = df["peak_ram_mb"].max() if "peak_ram_mb" in df.columns else None
    avg_cpu_eff = df["cpu_utilization_pct"].mean() if "cpu_utilization_pct" in df.columns else None

    st.markdown("## 🤖 Underlying Hardware Efficiency Profile")
    he_col1, he_col2, he_col3 = st.columns([2,2,2])
    with he_col1:
        if max_ram_spike is not None:
            st.metric(label="Max RAM Spike (MB)", value=f"{max_ram_spike:.2f} MB")
        else:
            st.info("`peak_ram_mb` column not found in DB.")
    with he_col2:
        if avg_cpu_eff is not None:
            st.metric(label="Avg CPU Utilization (%)", value=f"{avg_cpu_eff:.2f}%")
        else:
            st.info("`cpu_utilization_pct` column not found in DB.")
    with he_col3:
        samples = len(df) if ("peak_ram_mb" in df.columns or "cpu_utilization_pct" in df.columns) else 0
        st.metric(label="Hardware Samples", value=f"{samples}")

    st.markdown("---")

    # 4. Advanced System Performance Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📊 Token Efficiency Distribution")
        if "token_delta" in df.columns:
            st.bar_chart(data=df, y='token_delta', use_container_width=True)
        else:
            st.info("Token savings column missing from database schema.")

    with chart_col2:
        st.subheader("🎯 Semantic Guardrail Verification Correlation")
        if "token_delta" in df.columns and "bert_f1_score" in df.columns:
            st.scatter_chart(data=df, x='token_delta', y='bert_f1_score', use_container_width=True)
        else:
            st.info("Fidelity metrics missing from database schema.")

    st.markdown("---")

    # 5. Raw Ledger Inspection Matrix (Outputs all columns dynamically)
    st.subheader("📋 Proxy Transaction Infrastructure Log Ledger")
    st.dataframe(df, use_container_width=True)
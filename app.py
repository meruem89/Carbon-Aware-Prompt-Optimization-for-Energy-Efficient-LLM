from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.compression_engine import compress_prompt


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PRIMARY_DATABASE_PATH = DATA_DIR / "results.db"
FALLBACK_DATABASE_PATH = DATA_DIR / "benchmark_results.sqlite3"


st.set_page_config(
	page_title="Carbon-Aware Prompt Optimization Performance Telemetry",
	layout="wide",
	initial_sidebar_state="collapsed",
)

st.markdown(
	"""
	<style>
		.block-container {
			padding-top: 2rem;
			padding-bottom: 2rem;
		}
		.telemetry-subtitle {
			font-size: 1.05rem;
			color: #4b5563;
			margin-top: -0.5rem;
			margin-bottom: 1.5rem;
		}
		.metric-card {
			background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
			border: 1px solid #e5e7eb;
			border-radius: 16px;
			padding: 1rem 1.1rem;
			box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
		}
		.metric-label {
			font-size: 0.85rem;
			letter-spacing: 0.02em;
			text-transform: uppercase;
			color: #64748b;
			margin-bottom: 0.35rem;
		}
		.metric-value {
			font-size: 1.8rem;
			font-weight: 700;
			color: #0f172a;
			line-height: 1.1;
		}
		.metric-help {
			font-size: 0.8rem;
			color: #64748b;
			margin-top: 0.25rem;
		}
		.sandbox-panel {
			background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
			border: 1px solid #dbe4f0;
			border-radius: 18px;
			padding: 1.25rem;
			box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
		}
	</style>
	""",
	unsafe_allow_html=True,
)


def resolve_database_path() -> Path:
	if PRIMARY_DATABASE_PATH.exists():
		return PRIMARY_DATABASE_PATH
	if FALLBACK_DATABASE_PATH.exists():
		return FALLBACK_DATABASE_PATH
	return PRIMARY_DATABASE_PATH


def load_results_dataframe(database_path: Path) -> pd.DataFrame:
	with sqlite3.connect(database_path) as connection:
		return pd.read_sql_query("SELECT * FROM benchmark_results ORDER BY id ASC", connection)


def render_metric_card(label: str, value: str, help_text: str) -> None:
	st.markdown(
		f"""
		<div class="metric-card">
			<div class="metric-label">{label}</div>
			<div class="metric-value">{value}</div>
			<div class="metric-help">{help_text}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


st.title("Carbon-Aware Prompt Optimization Performance Telemetry")
st.markdown(
	"AMD hardware-aware optimization targets with benchmark telemetry for latency, cost, semantic fidelity, and carbon reduction.",
	unsafe_allow_html=False,
)

st.markdown('<div class="sandbox-panel">', unsafe_allow_html=True)
st.subheader("Claude Free Tier Token Saver Sandbox")
st.caption(
	"Paste a verbose prompt or codebase file, compress it for Claude Free Tier usage, and copy the optimized output instantly."
)

with st.form("claude_free_tier_token_saver_sandbox"):
	sandbox_input = st.text_area(
		"Prompt or codebase text",
		placeholder="Paste a long prompt, policy draft, README, or code file here...",
		height=240,
	)
	sandbox_target_rate = st.slider(
		"Target Compression Rate",
		min_value=0.3,
		max_value=0.8,
		value=0.5,
		step=0.05,
	)
	sandbox_submitted = st.form_submit_button("Compress Prompt for Claude")

if sandbox_submitted:
	if not sandbox_input.strip():
		st.warning("Paste a prompt or file contents before compressing.")
	else:
		with st.spinner("Compressing prompt for Claude Free Tier..."):
			sandbox_result = compress_prompt(sandbox_input, target_rate=sandbox_target_rate)
			original_tokens = sandbox_result["original_tokens"]
			compressed_tokens = sandbox_result["compressed_tokens"]
			tokens_saved = max(original_tokens - compressed_tokens, 0)
			tokens_saved_percentage = (tokens_saved / original_tokens * 100) if original_tokens else 0.0

			st.session_state["claude_sandbox_result"] = {
				"original_prompt": sandbox_input,
				"optimized_prompt": sandbox_result["compressed_prompt"],
				"original_tokens": original_tokens,
				"compressed_tokens": compressed_tokens,
				"tokens_saved": tokens_saved,
				"tokens_saved_percentage": tokens_saved_percentage,
			}

claude_sandbox_result = st.session_state.get("claude_sandbox_result")
if claude_sandbox_result:
	claude_columns = st.columns([1, 2], gap="large")
	with claude_columns[0]:
		render_metric_card(
			"Tokens Saved",
			f"{claude_sandbox_result['tokens_saved_percentage']:.1f}%",
			f"New expected token load: {claude_sandbox_result['compressed_tokens']:,} tokens",
		)
	with claude_columns[1]:
		st.text_area(
			"Optimized Claude Prompt",
			value=claude_sandbox_result["optimized_prompt"],
			height=240,
		)
		st.code(claude_sandbox_result["optimized_prompt"], language="markdown", copy_button=True)
		st.caption("Use the copy button above to move the optimized prompt into Claude immediately.")
else:
	st.info("Submit a prompt above to generate a compressed version for Claude Free Tier.")

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

database_path = resolve_database_path()
if not database_path.exists():
	st.error(
		"No SQLite database was found in data/results.db or data/benchmark_results.sqlite3. "
		"Run the benchmark pipeline first to generate telemetry."
	)
	st.stop()

results_frame = load_results_dataframe(database_path)

if results_frame.empty:
	st.warning("The benchmark_results table is empty. Run the benchmark script to populate telemetry.")
	st.stop()

for column_name in ["original_tokens", "compressed_tokens", "latency_saved_ms", "simulated_cost", "carbon_footprint_g", "bert_f1_score"]:
	if column_name in results_frame.columns:
		results_frame[column_name] = pd.to_numeric(results_frame[column_name], errors="coerce")

results_frame["original_tokens"] = results_frame["original_tokens"].fillna(0)
results_frame["compressed_tokens"] = results_frame["compressed_tokens"].fillna(0)
results_frame["latency_saved_ms"] = results_frame["latency_saved_ms"].fillna(0.0)
results_frame["simulated_cost"] = results_frame["simulated_cost"].fillna(0.0)
results_frame["carbon_footprint_g"] = results_frame["carbon_footprint_g"].fillna(0.0)
results_frame["bert_f1_score"] = results_frame["bert_f1_score"].fillna(0.0)

total_tokens_saved = int((results_frame["original_tokens"] - results_frame["compressed_tokens"]).clip(lower=0).sum())
average_latency_saved_ms = float(results_frame["latency_saved_ms"].mean())
total_simulated_cost_savings = float(results_frame["simulated_cost"].sum())
total_carbon_footprint_reduced = float(results_frame["carbon_footprint_g"].sum())

metric_columns = st.columns(4)
with metric_columns[0]:
	render_metric_card(
		"Total Tokens Saved",
		f"{total_tokens_saved:,}",
		"Aggregated token reduction across all benchmark runs.",
	)
with metric_columns[1]:
	render_metric_card(
		"Average Latency Saved (ms)",
		f"{average_latency_saved_ms:,.1f}",
		"Mean latency improvement from prompt compression.",
	)
with metric_columns[2]:
	render_metric_card(
		"Total Simulated Cost Savings ($)",
		f"${total_simulated_cost_savings:,.4f}",
		"Mock savings based on the benchmark token reduction model.",
	)
with metric_columns[3]:
	render_metric_card(
		"Total Carbon Footprint Reduced (g CO2e)",
		f"{total_carbon_footprint_reduced:,.4f}",
		"Carbon output reported by the telemetry pipeline.",
	)

st.divider()

figure_columns = st.columns(2)
with figure_columns[0]:
	st.subheader("Latency Saved by Benchmark Run")
	latency_bar = go.Figure(
		data=[
			go.Bar(
				x=results_frame["id"].astype(str) if "id" in results_frame.columns else results_frame.index.astype(str),
				y=results_frame["latency_saved_ms"],
				marker_color="#2563eb",
				hovertemplate="Run %{x}<br>Latency Saved: %{y:.2f} ms<extra></extra>",
			)
		]
	)
	latency_bar.update_layout(
		xaxis_title="Benchmark Run",
		yaxis_title="Latency Saved (ms)",
		template="plotly_white",
		height=440,
		margin=dict(l=20, r=20, t=20, b=20),
	)
	st.plotly_chart(latency_bar, width='stretch')

with figure_columns[1]:
	st.subheader("Carbon Footprint vs. BERTScore F1")
	carbon_scatter = go.Figure(
		data=[
			go.Scatter(
				x=results_frame["carbon_footprint_g"],
				y=results_frame["bert_f1_score"],
				mode="markers",
				marker=dict(
					size=12,
					color=results_frame["latency_saved_ms"],
					colorscale="Viridis",
					showscale=True,
					colorbar=dict(title="Latency Saved (ms)"),
				),
				text=results_frame["original_prompt"] if "original_prompt" in results_frame.columns else None,
				hovertemplate=(
					"Carbon: %{x:.4f} g CO2e<br>"
					"BERTScore F1: %{y:.4f}<br>"
					"<extra></extra>"
				),
			)
		]
	)
	carbon_scatter.add_hline(y=0.85, line_dash="dash", line_color="#dc2626", annotation_text="Semantic threshold = 0.85")
	carbon_scatter.update_layout(
		xaxis_title="Carbon Footprint (g CO2e)",
		yaxis_title="BERTScore F1",
		template="plotly_white",
		height=440,
		margin=dict(l=20, r=20, t=20, b=20),
	)
	st.plotly_chart(carbon_scatter, width='stretch')

st.divider()

st.subheader("Benchmark Results Explorer")
search_query = st.text_input(
	"Search the telemetry table",
	placeholder="Filter by prompt, category, response text, or metric value...",
)

display_frame = results_frame.copy()
if search_query.strip():
	query_lower = search_query.lower()
	mask = display_frame.astype(str).apply(
		lambda column: column.str.lower().str.contains(query_lower, na=False)
	).any(axis=1)
	display_frame = display_frame.loc[mask]

st.caption(f"Showing {len(display_frame)} of {len(results_frame)} rows from {database_path.name}")
st.dataframe(display_frame, width='stretch', height=520)

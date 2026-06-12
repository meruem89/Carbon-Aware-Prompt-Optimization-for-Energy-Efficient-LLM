
---

```markdown
# Carbon-Aware Prompt Optimization Engine on AMD Silicon

A hardware-aware, research-grade prompt compression proxy designed to exploit the mechanics of the AMD ROCm ecosystem. By intercepting verbose natural language payloads and applying high-speed neural token classification, this system systematically collapses input token sequences, drastically reducing VRAM consumption, pre-fill phase latency, operational costs, and localized carbon emissions.

---

## 1. System Architecture & Core Mechanics

In production LLM clusters running large-scale workloads (such as enterprise code-generation frameworks), natural language inputs introduce significant token redundancy. Because the self-attention mechanism in Transformer architectures possesses a time and space complexity of $O(N^2)$ (where $N$ is the sequence length), long prompts heavily penalize the compute-bound **pre-fill phase**, consuming immense High Bandwidth Memory (HBM) and maximizing GPU/CPU power draw.

This project sits as an abstract **API Preprocessing Middleware Proxy** between the application layer and the production inference cluster. 

```text
 [ User/Dev Application Payload ] 
                │
                ▼ (Sends High-Redundancy 5,000 Token Prompt)
 ┌────────────────────────────────────────────────────────┐
 │   COMPRESSION INTERMEDIARY PROXY (Our System)          │
 │                                                        │
 │   - Tiny 560M Bidirectional Encoder (LLMLingua-2)      │
 │   - High-Speed Binary Probability Masking (Keep/Drop)  │
 └────────────────────────────────────────────────────────┘
                │
                ▼ (Forwards 2,500 Dense, High-Information Tokens)
 [ Production LLM Serving Cluster (e.g., Llama-3-70B / Phi-3) ]

```

### The Multi-Model Parameter Asymmetry

To ensure the proxy does not introduce its own heavy computational bottlenecks, the framework utilizes **LLMLingua-2** (`xlm-roberta-large-meetingbank`).

* Unlike generative, auto-regressive LLMs that calculate token probabilities sequentially, LLMLingua-2 operates strictly as a **Token Classifier**.
* It reads the full bidirectional context in a single pass, mapping a binary probability mask (`Keep: 1` / `Drop: 0`) over the text.
* Running a **0.5 Billion parameter** encoder to shrink a prompt sequence before it hits a **70 Billion or 400 Billion parameter** inference cluster creates an exponential compute asymmetry: you expend a fraction of a penny in local CPU/edge compute to save massive financial and hardware energy expenditures on the primary cluster.

---

## 2. Directory Structure

```text
project/
│
├── data/
│   ├── prompts.csv         # Stratified 100-prompt baseline evaluation dataset
│   └── results.db          # Serverless SQLite telemetry repository
│
├── src/
│   ├── compression_engine.py  # LLMLingua-2 Token Classification implementation
│   ├── evaluator.py           # Contextual BERTScore fidelity validator
│   ├── carbon_estimator.py    # Hardware microjoule energy profiler (CodeCarbon)
│   ├── database.py            # SQLite schema initialization and persistence logic
│   └── run_benchmark.py       # Automated pipeline orchestration script
│
└── app.py                  # Streamlit analytical telemetry and sandbox UI

```

---

## 3. Engineering Metrics & Validation Guardrails

* **Token Metrics:** Absolute token counters mapped directly against the target model's Byte-Pair Encoding (BPE) tokenizer to guarantee mathematical precision over standard word-count estimations.
* **Semantic Fidelity Evaluation (BERTScore):** Traditional lexical string matching (BLEU/ROUGE) fails because LLMs naturally paraphrase accurate outputs. We enforce **BERTScore Contextual Embeddings** combined with `rescale_with_baseline=True` to normalize values. A strict safety guardrail threshold of **F1 > 0.85** ensures that overly aggressive token compression is dynamically caught and rejected before causing semantic collapse or LLM hallucinations.
* **Hardware Carbon Profiling:** Utilizing `CodeCarbon`'s `OfflineEmissionsTracker` to capture system microjoule power draw metrics. The tracker is explicitly calibrated to regional grid emission intensities (e.g., the Telangana grid index at ~0.683 kgCO₂/kWh) to output localized, empirically sound grams of $CO_2e$ performance calculations.

---

## 4. Local Deployment Checklist

Follow these step-by-step terminal instructions to stand up the pipeline and launch the web interface locally.

### Step 1: Initialize the Environment & Dependencies

Ensure you are running within the `project/` root directory. Build your virtual environment and install the required deep learning and telemetry tracking dependencies:

```powershell
# Navigate to the project directory
cd project

# Create the virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\activate

# Update pip and install libraries
python -m pip install --upgrade pip
pip install transformers torch sentence-transformers streamlit pandas plotly psutil llmlingua bert-score codecarbon

```

### Step 2: Execute the Benchmark Processing Pipeline

Run the automated orchestration engine to compress your prompt dataset, calculate hardware timing profiles, compute BERTScore contextual metrics, and persist the telemetry straight into SQLite:

```powershell
python src/run_benchmark.py

```

### Step 3: Launch the Telemetry Dashboard & Claude Sandbox

Launch the analytical frontend web application:

```powershell
streamlit run app.py

```

*Note: Due to Streamlit product specifications, layout visualization leverages the updated `width='stretch'` parameters to guarantee high-resolution canvas scaling across modern web browsers.*

---

## 5. Built-in Utility: Claude Free Tier Token Saver Sandbox

The top layer of the dashboard implements a highly practical, consumer-facing utility: **The Claude Free-Tier Token Saver**.

Commercial LLM platforms (like Anthropic's Claude free tier) do not restrict users by message counts, but by a rolling volume window of total tokens processed. When a user pastes a large prompt or a massive code module, Claude must re-read that entire context length for every subsequent question, instantly exhausting the account's free access allocation.

By passing lengthy scripts or inputs through our local, zero-cost token classifier first, structural boilerplate and syntactic fluff are completely stripped away on your laptop's CPU. Copy-pasting the resulting dense prompt into the Claude web interface cuts input sequence overhead by **40% to 60%**, effectively doubling or tripling your rolling interaction limits from 2 questions to 5+ clean contextual exchanges.

```

```
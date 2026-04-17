# AIPE Assignment 4 — Airline Pricing & Marketing Mini-Agent (LangGraph)

This project builds a mini agent-like pipeline using **LangGraph** with:

- **3+ nodes** (this project has 5)
- **2+ LLM calls** using **Pydantic Structured Outputs** (this project uses 3)
- **Pydantic state + models**
- All prompts stored in `prompts.py` as named constants

## What it does

Given synthetic airline data (flights, sales signals, competitor prices), the pipeline:

1. Loads and validates input tables (code).
2. Uses an LLM to extract key metrics, segments, trends, and data-quality flags (structured output).
3. Uses an LLM to propose route/class pricing & marketing actions (structured output).
4. Runs a simple revenue/demand simulation to validate decisions (code).
5. Uses an LLM to produce an executive-ready final strategy report (structured output).

## Project structure

```
aipe-airline-agent/
├── main.py          — runs the LangGraph pipeline end-to-end
├── models.py        — Pydantic state + structured output schemas
├── prompts.py       — prompts (named constants)
├── make_data.py     — generates synthetic CSV inputs into data/
├── data/            — input CSV files (flights.csv, competitors.csv)
└── outputs/         — saved JSON outputs per run
```

## Setup (step-by-step)

### 1) Create and activate a virtual environment (recommended)

```bash
cd "/Users/erasylnurlanbai/Documents/aipe-airline-agent"
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Choose an LLM provider and set API key

This project supports **Groq** (recommended) or **Gemini**.

#### Option A: Groq (recommended — free tier)

```bash
export LLM_PROVIDER="groq"
export GROQ_API_KEY="YOUR_KEY_HERE"
```

#### Option B: Gemini

```bash
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

You can also put them in a `.env` file (see `.env.example`).

### 4) Generate synthetic input data

```bash
python make_data.py
```

This creates `data/flights.csv` and `data/competitors.csv`.

### 5) Run the pipeline

```bash
python main.py
```

Outputs are saved into `outputs/` as JSON files named by `run_id`:

```
outputs/
├── <run_id>_parsed_signals.json
├── <run_id>_pricing_plan.json
├── <run_id>_simulation.json
└── <run_id>_exec_report.json
```

## Pipeline nodes

| Step | Node | Type | Description |
|------|------|------|-------------|
| 1/6 | `load_inputs` | Code | Reads CSVs, validates with Pydantic |
| 2/6 | `parse_signals_node` | LLM | Extracts metrics, segments, trends |
| 3/6 | `pricing_plan_node` | LLM | Proposes pricing & marketing actions |
| 4/6 | `simulate_node` | Code | Simulates revenue/demand impact |
| 5/6 | `exec_report_node` | LLM | Generates executive strategy report |
| 6/6 | `export` | Code | Saves all artifacts as JSON |

## Implementation notes

- **LangGraph state:** `app.invoke()` returns `AddableValuesDict`, not `AgentState`. The `_state_from_dict()` helper safely converts it back to `AgentState` so attribute access works correctly.
- **Structured JSON outputs:** All LLM calls enforce strict JSON schemas via prompt instructions. Responses are parsed and validated with Pydantic.
- **Simulation:** The revenue/demand simulator is intentionally simple (mini-project scope). It uses price elasticity per cabin class and a competitor price gap penalty to produce KPI estimates.
- **Enum safety:** All LLM-generated enum fields (`action`, `cabin_class`, `confidence`, etc.) are constrained in the prompt and validated by Pydantic on parse.
from __future__ import annotations

import json
import os
import uuid
from datetime import date
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph

from models import (
    AgentState,
    CompetitorRow,
    ExecReport,
    FlightRow,
    ParsedSignals,
    PricingPlan,
    SimulationResult,
    SimulationSummary,
    SimResultPerDecision,
)
from prompts import (
    NODE1_PARSE_PROMPT,
    NODE2_PRICING_ANALYSIS_PROMPT,
    NODE3_STRATEGY_OUTPUT_PROMPT,
    SYSTEM_PROMPT,
)


def build_llm() -> Any:
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(api_key=api_key, model=model, temperature=0.2)


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def node0_load_inputs(state: AgentState) -> dict:
    print("[1/5] load_inputs: reading CSVs...")
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    flights_df = pd.read_csv(os.path.join(data_dir, "flights.csv"))
    comp_df = pd.read_csv(os.path.join(data_dir, "competitors.csv"))

    flights = [FlightRow.model_validate(r) for r in flights_df.to_dict(orient="records")]
    competitors = [CompetitorRow.model_validate(r) for r in comp_df.to_dict(orient="records")]

    run_id = state.run_id or uuid.uuid4().hex[:10]
    output_dir = os.getenv("OUTPUT_DIR", state.output_dir or "outputs")
    _ensure_output_dir(output_dir)

    return {"flights": flights, "competitors": competitors, "run_id": run_id, "output_dir": output_dir}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _compact_dataset_for_llm(state: AgentState | dict, max_rows: int = 20) -> dict[str, Any]:
    # Support both AgentState object and plain dict (from LangGraph invoke result)
    if isinstance(state, dict):
        flights = state.get("flights", [])
        competitors = state.get("competitors", [])
    else:
        flights = state.flights
        competitors = state.competitors

    flights_df = pd.DataFrame([f.model_dump() for f in flights])

    for col in flights_df.columns:
        if flights_df[col].dtype == "object":
            flights_df[col] = flights_df[col].astype(str)

    flights_df["load_factor"] = (flights_df["seats_sold"] / flights_df["seats_total"]).round(3)

    sampled = (
        flights_df.sort_values(["route", "cabin_class", "days_to_departure"])
        .groupby(["route", "cabin_class"], as_index=False)
        .head(1)
    )
    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=42)

    comp_df = pd.DataFrame([c.model_dump() for c in competitors])
    comp_summary = (
        comp_df.groupby(["route", "cabin_class"])["competitor_price"]
        .agg(min_price="min", avg_price="mean", max_price="max")
        .reset_index()
    )

    return {
        "flights_sample": sampled.to_dict(orient="records"),
        "competitor_price_summary": comp_summary.to_dict(orient="records"),
        "notes": {"load_factor": "seats_sold / seats_total"},
    }


def _invoke_json(llm: Any, system_prompt: str, user_prompt: str) -> dict:
    """Call LLM and parse JSON response."""
    from langchain_core.messages import HumanMessage, SystemMessage

    full_prompt = (
        user_prompt
        + "\n\nIMPORTANT: Respond ONLY with a valid JSON object. "
        "No markdown, no explanation, no extra text. Start with { and end with }."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=full_prompt),
    ])

    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def _state_from_dict(d: dict) -> AgentState:
    """Safely reconstruct AgentState from a LangGraph result dict."""
    return AgentState(**{k: v for k, v in d.items() if k in AgentState.model_fields})


def node1_parse_signals(llm: Any) -> Callable[[AgentState], dict]:
    def _run(state: AgentState) -> dict:
        print("[2/5] parse_signals_node: calling LLM...")
        compact = _compact_dataset_for_llm(state)

        schema_hint = """{
  "key_metrics": ["load_factor per route", "price_gap vs competitor"],
  "segments": ["business travelers", "leisure"],
  "notable_trends": ["ALA-DXB high demand"],
  "data_quality_flags": ["no missing values detected"],
  "assumptions": ["demand index is synthetic"]
}"""

        user_prompt = (
            f"{NODE1_PARSE_PROMPT}\n\n"
            f"INPUT_DATA:\n{json.dumps(compact, ensure_ascii=False, default=_json_default)}\n\n"
            f"Return JSON matching this structure:\n{schema_hint}"
        )

        data = _invoke_json(llm, SYSTEM_PROMPT, user_prompt)

        for field in ["key_metrics", "segments", "notable_trends", "data_quality_flags", "assumptions"]:
            val = data.get(field, [])
            if isinstance(val, str):
                data[field] = [val]
            elif not isinstance(val, list):
                data[field] = [str(val)]

        parsed = ParsedSignals.model_validate(data)
        return {"parsed_signals": parsed}

    return _run


def node2_pricing_plan(llm: Any) -> Callable[[AgentState], dict]:
    def _run(state: AgentState) -> dict:
        print("[3/5] pricing_plan_node: calling LLM...")
        compact = _compact_dataset_for_llm(state)

        schema_hint = """{
  "decisions": [
    {
      "route": "ALA-AST",
      "cabin_class": "economy",
      "action": "raise",
      "price_change_pct": 5.0,
      "target_segment": "leisure",
      "promo_type": "none",
      "rationale": "high demand, low load factor",
      "constraints": ["max +10%"],
      "confidence": "high"
    }
  ],
  "global_notes": ["seasonal demand is high"],
  "assumptions": ["competitor prices are stable"]
}

STRICT ENUM VALUES — use exactly these strings:
- action: "raise" | "lower" | "hold"
- cabin_class: "economy" | "business" | "first"
- target_segment: "business" | "leisure" | "student" | "family" | "other"
- promo_type: "none" | "flash_sale" | "promo_code" | "bundle" | "loyalty_bonus" | "student_discount"
- confidence: "high" | "medium" | "low"
- price_change_pct: float (e.g. 5.0 for +5%, -3.0 for -3%, 0.0 for hold)"""

        user_prompt = (
            f"{NODE2_PRICING_ANALYSIS_PROMPT}\n\n"
            f"PARSED_SIGNALS:\n{state.parsed_signals.model_dump_json()}\n\n"
            f"INPUT_DATA:\n{json.dumps(compact, ensure_ascii=False, default=_json_default)}\n\n"
            f"Return JSON matching this structure:\n{schema_hint}"
        )

        data = _invoke_json(llm, SYSTEM_PROMPT, user_prompt)

        for d in data.get("decisions", []):
            if not isinstance(d.get("constraints"), list):
                d["constraints"] = []
        if not isinstance(data.get("global_notes"), list):
            data["global_notes"] = []
        if not isinstance(data.get("assumptions"), list):
            data["assumptions"] = []

        plan = PricingPlan.model_validate(data)
        return {"pricing_plan": plan}

    return _run


def node3_simulate(state: AgentState) -> dict:
    print("[4/5] simulate_node: running simulation...")
    if not state.pricing_plan:
        raise ValueError("pricing_plan is missing")

    flights_df = pd.DataFrame([f.model_dump() for f in state.flights])
    flights_df["load_factor"] = flights_df["seats_sold"] / flights_df["seats_total"]

    base_by_route_class = flights_df.groupby(["route", "cabin_class"], as_index=False).agg(
        base_price=("base_price", "mean"),
        current_load_factor=("load_factor", "mean"),
        demand_index=("historical_demand_index", "mean"),
        velocity=("recent_booking_velocity", "mean"),
    )

    comp_df = pd.DataFrame([c.model_dump() for c in state.competitors])
    comp_agg = (
        comp_df.groupby(["route", "cabin_class"], as_index=False)["competitor_price"]
        .mean()
        .rename(columns={"competitor_price": "competitor_avg_price"})
    )

    joined = base_by_route_class.merge(comp_agg, on=["route", "cabin_class"], how="left")
    elasticity = {"economy": 1.25, "business": 0.85, "first": 0.60}

    per: list[SimResultPerDecision] = []
    high_risk = 0

    for d in state.pricing_plan.decisions:
        row = joined[(joined["route"] == d.route) & (joined["cabin_class"] == d.cabin_class)]
        if row.empty:
            per.append(SimResultPerDecision(
                route=d.route, cabin_class=d.cabin_class,
                base_price=0.0, new_price=0.0,
                current_load_factor=0.0, expected_load_factor=0.0,
                expected_revenue_change_pct=0.0,
                risk_flags=["missing_route_class_in_data"],
            ))
            high_risk += 1
            continue

        r0 = row.iloc[0]
        base_price = float(r0["base_price"])
        current_lf = float(r0["current_load_factor"])
        demand_index = float(r0["demand_index"])
        velocity = float(r0["velocity"])
        comp_price = float(r0["competitor_avg_price"]) if pd.notna(r0["competitor_avg_price"]) else base_price

        pct = float(d.price_change_pct) / 100.0
        new_price = base_price * (1.0 + pct)
        strength = float(max(0.7, min(1.5, 0.65 * demand_index + 0.35 * velocity)))
        gap = (new_price - comp_price) / max(comp_price, 1.0)
        e = elasticity[d.cabin_class]
        delta = (-e * pct) * (1.0 / strength) - 0.18 * max(0.0, gap)
        expected_lf = float(max(0.05, min(0.98, current_lf + delta)))
        rev_change_pct = 100.0 * ((new_price * expected_lf) / max(base_price * current_lf, 1e-6) - 1.0)

        risk_flags: list[str] = []
        if abs(d.price_change_pct) > 18:
            risk_flags.append("large_price_change")
        if expected_lf < current_lf - 0.12 and d.action.value == "raise":
            risk_flags.append("demand_drop_risk")
        if gap > 0.10:
            risk_flags.append("above_competitor_risk")
        if d.confidence == "low":
            risk_flags.append("low_confidence")
        if any(x in risk_flags for x in ("demand_drop_risk", "above_competitor_risk", "large_price_change")):
            high_risk += 1

        per.append(SimResultPerDecision(
            route=d.route, cabin_class=d.cabin_class,
            base_price=round(base_price, 2), new_price=round(new_price, 2),
            current_load_factor=round(current_lf, 3), expected_load_factor=round(expected_lf, 3),
            expected_revenue_change_pct=round(rev_change_pct, 2),
            risk_flags=risk_flags,
        ))

    rev_changes = [x.expected_revenue_change_pct for x in per if x.base_price > 0]
    total_change = float(sum(rev_changes) / max(len(rev_changes), 1))

    sim = SimulationResult(
        per_decision=per,
        summary=SimulationSummary(
            expected_total_revenue_change_pct=round(total_change, 2),
            high_risk_count=high_risk,
            notes=["Simulation is synthetic. Use as KPI estimate only."],
        ),
    )
    return {"simulation_result": sim}


def node4_exec_report(llm: Any) -> Callable[[AgentState], dict]:
    def _run(state: AgentState) -> dict:
        print("[5/5] exec_report_node: calling LLM...")
        if not state.parsed_signals or not state.pricing_plan or not state.simulation_result:
            raise ValueError("Missing required artifacts for exec report")

        compact = _compact_dataset_for_llm(state)

        schema_hint = """{
  "executive_summary": ["point 1", "point 2"],
  "pricing_recommendations": ["raise ALA-AST economy by 5%", "hold ALA-DXB business"],
  "route_class_priority_ranking": ["1. ALA-DXB business", "2. ALA-AST economy"],
  "competitive_advantage_ideas": ["idea 1", "idea 2"],
  "risk_analysis": ["risk 1", "risk 2"],
  "next_actions": ["action 1", "action 2"],
  "expected_impact": {
    "expected_total_revenue_change_pct": 3.5,
    "expected_load_factor_change_pct": 1.2,
    "high_risk_count": 1
  },
  "assumptions": ["assumption 1"]
}

CRITICAL: Every top-level field MUST be a JSON array of strings, NOT a plain string.
executive_summary, pricing_recommendations, route_class_priority_ranking,
competitive_advantage_ideas, risk_analysis, next_actions, assumptions
— all must be arrays like ["item1", "item2"]."""

        user_prompt = (
            f"{NODE3_STRATEGY_OUTPUT_PROMPT}\n\n"
            f"PARSED_SIGNALS:\n{state.parsed_signals.model_dump_json()}\n\n"
            f"PRICING_PLAN:\n{state.pricing_plan.model_dump_json()}\n\n"
            f"SIMULATION_RESULT:\n{state.simulation_result.model_dump_json()}\n\n"
            f"INPUT_DATA:\n{json.dumps(compact, ensure_ascii=False, default=_json_default)}\n\n"
            f"Return JSON matching this structure exactly:\n{schema_hint}"
        )

        data = _invoke_json(llm, SYSTEM_PROMPT, user_prompt)

        list_fields = [
            "executive_summary", "pricing_recommendations",
            "route_class_priority_ranking", "competitive_advantage_ideas",
            "risk_analysis", "next_actions", "assumptions",
        ]
        for field in list_fields:
            val = data.get(field, [])
            if isinstance(val, str):
                data[field] = [val]
            elif not isinstance(val, list):
                data[field] = [str(val)]

        ei = data.get("expected_impact", {})
        if not isinstance(ei, dict):
            ei = {}
        sim_summary = state.simulation_result.summary
        data["expected_impact"] = {
            "expected_total_revenue_change_pct": float(ei.get(
                "expected_total_revenue_change_pct",
                sim_summary.expected_total_revenue_change_pct
            )),
            "expected_load_factor_change_pct": float(ei.get("expected_load_factor_change_pct", 0.0)),
            "high_risk_count": int(ei.get("high_risk_count", sim_summary.high_risk_count)),
        }

        report = ExecReport.model_validate(data)
        return {"exec_report": report}

    return _run


def _save_json(output_dir: str, filename: str, payload: Any) -> str:
    _ensure_output_dir(output_dir)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def node5_export(state: AgentState) -> dict:
    print("[6/6] export: saving JSON outputs...")
    out_dir = state.output_dir
    run_id = state.run_id

    _save_json(out_dir, f"{run_id}_parsed_signals.json", state.parsed_signals.model_dump())
    _save_json(out_dir, f"{run_id}_pricing_plan.json", state.pricing_plan.model_dump())
    _save_json(out_dir, f"{run_id}_simulation.json", state.simulation_result.model_dump())
    _save_json(out_dir, f"{run_id}_exec_report.json", state.exec_report.model_dump())
    return {}


def build_graph(llm: Any):
    g = StateGraph(AgentState)

    g.add_node("load_inputs", node0_load_inputs)
    g.add_node("parse_signals_node", node1_parse_signals(llm))
    g.add_node("pricing_plan_node", node2_pricing_plan(llm))
    g.add_node("simulate_node", node3_simulate)
    g.add_node("exec_report_node", node4_exec_report(llm))
    g.add_node("export", node5_export)

    g.set_entry_point("load_inputs")
    g.add_edge("load_inputs", "parse_signals_node")
    g.add_edge("parse_signals_node", "pricing_plan_node")
    g.add_edge("pricing_plan_node", "simulate_node")
    g.add_edge("simulate_node", "exec_report_node")
    g.add_edge("exec_report_node", "export")
    g.set_finish_point("export")

    return g.compile()


def main() -> None:
    load_dotenv()

    # ✅ LangSmith tracing — осы жерге қос
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING_V2", "true"))
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", "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"))
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "aipe-airline-agent"))
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))

    llm = build_llm()
    app = build_graph(llm)

    initial_state = AgentState()
    print("Starting pipeline...")

    result: dict = app.invoke(initial_state)
    final_state = _state_from_dict(result)

    print("\nDone!")
    print(f"  run_id : {final_state.run_id}")
    print(f"  outputs: {os.path.abspath(final_state.output_dir)}")


if __name__ == "__main__":
    main()
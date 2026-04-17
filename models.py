from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


CabinClass = Literal["economy", "business", "first"]
Segment = Literal["business", "leisure", "student", "family", "other"]
PromoType = Literal["none", "flash_sale", "promo_code", "bundle", "loyalty_bonus", "student_discount"]


class FlightRow(BaseModel):
    flight_id: str
    route: str  # e.g. "ALA-AST"
    departure_date: date
    cabin_class: CabinClass
    seats_total: int = Field(ge=1)
    seats_sold: int = Field(ge=0)
    base_price: float = Field(ge=0)
    season: str  # e.g. "winter", "spring", "summer", "autumn"
    days_to_departure: int = Field(ge=0)
    historical_demand_index: float = Field(ge=0)  # synthetic index
    recent_booking_velocity: float = Field(ge=0)  # synthetic index


class CompetitorRow(BaseModel):
    route: str
    cabin_class: CabinClass
    competitor: str
    competitor_price: float = Field(ge=0)


class ParsedSignals(BaseModel):
    key_metrics: list[str] = Field(description="Main derived metrics to use downstream.")
    segments: list[str] = Field(description="Relevant customer segments inferred from data.")
    notable_trends: list[str] = Field(description="High-signal trends and patterns.")
    data_quality_flags: list[str] = Field(description="Missing/suspicious values or limitations.")
    assumptions: list[str] = Field(description="Explicit assumptions if something is unknown.")


class PricingAction(str, Enum):
    raise_price = "raise"
    lower_price = "lower"
    hold_price = "hold"


class RouteClassDecision(BaseModel):
    route: str
    cabin_class: CabinClass
    action: PricingAction
    price_change_pct: float = Field(description="Percent change vs current base price. Example: 5.0 means +5%.")
    target_segment: Segment
    promo_type: PromoType
    rationale: str
    constraints: list[str] = Field(default_factory=list, description="Guardrails, e.g. min/max price, avoid demand loss.")
    confidence: Literal["high", "medium", "low"]


class PricingPlan(BaseModel):
    decisions: list[RouteClassDecision]
    global_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class SimResultPerDecision(BaseModel):
    route: str
    cabin_class: CabinClass
    base_price: float
    new_price: float
    current_load_factor: float
    expected_load_factor: float
    expected_revenue_change_pct: float
    risk_flags: list[str] = Field(default_factory=list)


class SimulationSummary(BaseModel):
    expected_total_revenue_change_pct: float
    high_risk_count: int
    notes: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    per_decision: list[SimResultPerDecision]
    summary: SimulationSummary


class ExpectedImpact(BaseModel):
    expected_total_revenue_change_pct: float
    expected_load_factor_change_pct: float
    high_risk_count: int


class ExecReport(BaseModel):
    executive_summary: list[str]
    pricing_recommendations: list[str]
    route_class_priority_ranking: list[str]
    competitive_advantage_ideas: list[str]
    risk_analysis: list[str]
    next_actions: list[str]
    expected_impact: ExpectedImpact
    assumptions: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    # Inputs
    flights: list[FlightRow] = Field(default_factory=list)
    competitors: list[CompetitorRow] = Field(default_factory=list)

    # LLM outputs
    parsed_signals: ParsedSignals | None = None
    pricing_plan: PricingPlan | None = None
    exec_report: ExecReport | None = None

    # Code outputs
    simulation_result: SimulationResult | None = None

    # Metadata
    output_dir: str = "outputs"
    run_id: str = ""


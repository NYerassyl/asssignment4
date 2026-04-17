from __future__ import annotations

import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def _season_for_month(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def main() -> None:
    load_dotenv()
    seed = int(os.getenv("RANDOM_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    routes = [
        "ALA-AST",
        "ALA-SCO",
        "ALA-DXB",
        "AST-IST",
        "AST-ALA",
        "SCO-ALA",
        "ALA-TSE",
        "TSE-ALA",
    ]
    cabin_classes = ["economy", "business", "first"]
    competitors = ["SkyJet", "NomadAir", "EurasiaFly"]

    today = date.today()
    flights_rows: list[dict] = []

    # Generate ~ 8 routes * 3 cabins * 6 departure dates = 144 rows
    departure_offsets = [7, 14, 21, 30, 45, 60]
    flight_idx = 1
    for route in routes:
        for cabin in cabin_classes:
            for d in departure_offsets:
                dep = today + timedelta(days=d)
                season = _season_for_month(dep.month)

                seats_total = {"economy": 180, "business": 28, "first": 8}[cabin]
                # synthetic seat sales intensity: depends on cabin + route popularity + days-to-dep
                route_pop = {
                    "ALA-DXB": 1.25,
                    "AST-IST": 1.20,
                    "ALA-AST": 1.10,
                }.get(route, 1.0)
                time_factor = 1.0 + (30 - min(d, 30)) / 60.0  # closer date => higher
                cabin_factor = {"economy": 1.0, "business": 0.75, "first": 0.55}[cabin]

                base_price = {
                    "economy": random.uniform(180, 420),
                    "business": random.uniform(700, 1400),
                    "first": random.uniform(1600, 2800),
                }[cabin]
                # seasonal adjustment
                if season == "summer":
                    base_price *= 1.08
                elif season == "winter":
                    base_price *= 0.97

                demand_index = float(np.clip(np.random.normal(1.0 * route_pop, 0.15), 0.5, 1.6))
                velocity = float(np.clip(np.random.normal(1.0 * time_factor, 0.25), 0.4, 1.9))

                expected_load = np.clip(0.35 + 0.35 * demand_index + 0.15 * velocity, 0.10, 0.98)
                seats_sold = int(np.clip(np.random.normal(expected_load, 0.08), 0.05, 0.99) * seats_total)

                flights_rows.append(
                    {
                        "flight_id": f"F{flight_idx:04d}",
                        "route": route,
                        "departure_date": dep.isoformat(),
                        "cabin_class": cabin,
                        "seats_total": seats_total,
                        "seats_sold": seats_sold,
                        "base_price": round(base_price, 2),
                        "season": season,
                        "days_to_departure": d,
                        "historical_demand_index": round(demand_index, 3),
                        "recent_booking_velocity": round(velocity, 3),
                    }
                )
                flight_idx += 1

    flights_df = pd.DataFrame(flights_rows)
    flights_path = os.path.join(out_dir, "flights.csv")
    flights_df.to_csv(flights_path, index=False)

    # Competitor prices: competitor_price around our base_price with noise and competitor bias
    comp_rows: list[dict] = []
    # Take a representative base_price per route/class for competitor table
    rep = (
        flights_df.groupby(["route", "cabin_class"], as_index=False)["base_price"]
        .mean()
        .rename(columns={"base_price": "our_avg_price"})
    )
    for _, r in rep.iterrows():
        for comp in competitors:
            bias = {"SkyJet": -0.03, "NomadAir": 0.00, "EurasiaFly": 0.04}[comp]
            noise = float(np.random.normal(0.0, 0.04))
            comp_price = float(r["our_avg_price"]) * (1.0 + bias + noise)
            comp_rows.append(
                {
                    "route": r["route"],
                    "cabin_class": r["cabin_class"],
                    "competitor": comp,
                    "competitor_price": round(max(1.0, comp_price), 2),
                }
            )
    comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(out_dir, "competitors.csv")
    comp_df.to_csv(comp_path, index=False)

    print("Generated:")
    print(f"- {flights_path}")
    print(f"- {comp_path}")


if __name__ == "__main__":
    main()


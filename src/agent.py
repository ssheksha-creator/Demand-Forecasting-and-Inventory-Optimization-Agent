from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# In the real project this file lives in src/, so parent.parent is the project root.
# The fallback also makes the file testable when temporarily placed beside the CSV.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "sales_data.csv").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent

ROOT = PROJECT_ROOT
DATA_PATH = ROOT / "sales_data.csv"
REC_PATH = ROOT / "outputs" / "recommendations_v2.csv"
FC_PATH = ROOT / "outputs" / "forecasts_v2.csv"


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
def _load_sales() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _load_recommendations() -> pd.DataFrame:
    return pd.read_csv(REC_PATH) if REC_PATH.exists() else pd.DataFrame()


def _load_forecasts() -> pd.DataFrame:
    if not FC_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FC_PATH)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def _col(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _records(df: pd.DataFrame, columns: List[str] | None = None, limit: int = 100):
    if df is None or df.empty:
        return []
    if columns is not None:
        cols = [c for c in columns if c in df.columns]
        x = df[cols].copy()
    else:
        x = df.copy()
    return x.head(limit).replace({np.nan: None}).to_dict("records")


def _fmt(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "N/A"


def _extract_store_ids(q: str) -> List[str]:
    q = q.upper()
    found = re.findall(r"\bS\d{3}\b", q)
    # Also understand natural-language forms such as "store 1", "store1".
    found += [f"S{int(x):03d}" for x in re.findall(r"\bSTORE\s*0*(\d+)\b", q)]
    return list(dict.fromkeys(found))


def _extract_product_ids(q: str) -> List[str]:
    q = q.upper()
    found = re.findall(r"\bP\d{4}\b", q)
    found += [f"P{int(x):04d}" for x in re.findall(r"\bPRODUCT\s*0*(\d+)\b", q)]
    return list(dict.fromkeys(found))


def _extract_percent(q: str, default: float = 20) -> float:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", q)
    return float(m.group(1)) if m else default


def _safe_num(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ---------------------------------------------------------------------
# Chart builders. The agent decides which visuals are useful.
# app.py only renders these specifications.
# ---------------------------------------------------------------------
def _line_chart(title: str, data: list, x: str, series: list) -> dict:
    return {"type": "line", "title": title, "x": x, "series": series, "data": data}


def _bar_chart(title: str, data: list, x: str, series: list) -> dict:
    return {"type": "bar", "title": title, "x": x, "series": series, "data": data}


# ---------------------------------------------------------------------
# Product comparison
# ---------------------------------------------------------------------
def compare_products(question: str, store_id: str, products: List[str]) -> Dict[str, Any]:
    products = list(dict.fromkeys(products))[:2]
    if len(products) < 2:
        return {"error": "Please specify two products, for example: Compare P0018 and P0016 in S002."}

    df = _load_sales()
    sdf = df[
        (df["Store ID"].astype(str).str.upper() == store_id.upper())
        & (df["Product ID"].astype(str).str.upper().isin(products))
    ].copy()

    if sdf.empty:
        return {"error": f"No records were found for {store_id} and {', '.join(products)}."}

    summary = sdf.groupby("Product ID", as_index=False).agg(
        Historical_Avg_Demand=("Demand", "mean"),
        Total_Historical_Demand=("Demand", "sum"),
        Historical_Demand_Std=("Demand", "std"),
        Average_Inventory=("Inventory Level", "mean"),
        Average_Units_Sold=("Units Sold", "mean"),
        Average_Units_Ordered=("Units Ordered", "mean"),
        Average_Price=("Price", "mean"),
        Average_Discount=("Discount", "mean"),
        Promotion_Rate=("Promotion", lambda x: x.astype(str).str.lower().isin(["yes", "true", "1"]).mean() * 100),
    )

    rec = _load_recommendations()
    if not rec.empty:
        r = rec[
            (rec["Store ID"].astype(str).str.upper() == store_id.upper())
            & (rec["Product ID"].astype(str).str.upper().isin(products))
        ].copy()
        rcols = [
            "Product ID", "Current Inventory", "Forecast P50 Demand",
            "Forecast P90 Demand", "Safety Stock", "Reorder Point",
            "Target Inventory", "Recommended Order", "Projected Shortage",
            "Days Until Stockout", "Inventory Coverage Days", "Stockout Risk",
            "Risk Score", "Risk Level", "Action Priority"
        ]
        summary = summary.merge(
            r[[c for c in rcols if c in r.columns]],
            on="Product ID",
            how="left",
        )

    # Recent 30-day evidence.
    recent = (
        sdf.sort_values("Date")
        .groupby("Product ID", group_keys=False)
        .tail(30)
    )
    recent_stats = recent.groupby("Product ID", as_index=False).agg(
        Recent_Avg_Demand=("Demand", "mean"),
        Recent_Avg_Inventory=("Inventory Level", "mean"),
    )
    summary = summary.merge(recent_stats, on="Product ID", how="left")

    # Keep requested product order.
    summary["__order"] = summary["Product ID"].map({p: i for i, p in enumerate(products)})
    summary = summary.sort_values("__order").drop(columns="__order")

    # Historical time-series chart: last 120 days for EACH product.
    hist_parts = []
    for p in products:
        hist_parts.append(
            sdf[sdf["Product ID"].astype(str).str.upper() == p]
            .sort_values("Date")
            .tail(120)[["Date", "Demand", "Product ID"]]
        )
    hist = pd.concat(hist_parts, ignore_index=True)
    line = (
        hist.pivot_table(index="Date", columns="Product ID", values="Demand", aggfunc="mean")
        .reset_index()
        .sort_values("Date")
    )
    line["Date"] = line["Date"].dt.strftime("%Y-%m-%d")
    line_records = line.to_dict("records")

    # Comparison metrics chart.
    bars = []
    for _, row in summary.iterrows():
        bars.append({
            "Product": row["Product ID"],
            "Current Inventory": row.get("Current Inventory", row.get("Average_Inventory")),
            "P50 Forecast": row.get("Forecast P50 Demand"),
            "P90 Forecast": row.get("Forecast P90 Demand"),
            "Recommended Order": row.get("Recommended Order"),
        })

    # Driver table.
    factors = summary[[
        "Product ID", "Historical_Avg_Demand", "Recent_Avg_Demand",
        "Average_Inventory", "Average_Price", "Average_Discount",
        "Promotion_Rate", "Historical_Demand_Std"
    ]].copy()

    a = summary.iloc[0]
    b = summary.iloc[1]
    p1, p2 = products

    reasons: List[str] = []

    # Demand level and recent trend.
    d1, d2 = _safe_num(a["Historical_Avg_Demand"]), _safe_num(b["Historical_Avg_Demand"])
    if abs(d1 - d2) >= 0.01 * max(d1, d2, 1):
        higher = p1 if d1 > d2 else p2
        reasons.append(
            f"{higher} has higher historical demand at {_fmt(max(d1, d2))} units/day "
            f"versus {_fmt(min(d1, d2))} for the other product."
        )

    rd1, rd2 = _safe_num(a["Recent_Avg_Demand"]), _safe_num(b["Recent_Avg_Demand"])
    if abs(rd1 - rd2) >= 0.01 * max(rd1, rd2, 1):
        higher = p1 if rd1 > rd2 else p2
        reasons.append(
            f"Recent demand also favors {higher}: {_fmt(max(rd1, rd2))} units/day "
            f"versus {_fmt(min(rd1, rd2))} over the latest 30 observations."
        )

    # Coverage / stockout pressure.
    i1, i2 = a.get("Current Inventory"), b.get("Current Inventory")
    f1, f2 = a.get("Forecast P50 Demand"), b.get("Forecast P50 Demand")
    if pd.notna(i1) and pd.notna(i2) and pd.notna(f1) and pd.notna(f2):
        cov1 = _safe_num(i1) / max(_safe_num(f1), 1) * 7
        cov2 = _safe_num(i2) / max(_safe_num(f2), 1) * 7
        tighter = p1 if cov1 < cov2 else p2
        reasons.append(
            f"{tighter} has tighter forecast coverage ({_fmt(min(cov1, cov2), 2)} days "
            f"versus {_fmt(max(cov1, cov2), 2)} days), so its current stock is less adequate "
            "relative to expected demand."
        )

    days1, days2 = a.get("Days Until Stockout"), b.get("Days Until Stockout")
    if pd.notna(days1) and pd.notna(days2) and _safe_num(days1) != _safe_num(days2):
        sooner = p1 if _safe_num(days1) < _safe_num(days2) else p2
        reasons.append(
            f"{sooner} reaches projected stockout sooner ({_fmt(min(_safe_num(days1), _safe_num(days2)), 2)} "
            f"days versus {_fmt(max(_safe_num(days1), _safe_num(days2)), 2)} days)."
        )

    # Pricing/promotion differences are descriptive associations only.
    dis1, dis2 = _safe_num(a["Average_Discount"]), _safe_num(b["Average_Discount"])
    if abs(dis1 - dis2) >= 2:
        higher = p1 if dis1 > dis2 else p2
        reasons.append(
            f"{higher} had the higher historical average discount "
            f"({_fmt(max(dis1, dis2))}% vs {_fmt(min(dis1, dis2))}%). "
            "This is an observed association, not proof of causation."
        )

    promo1, promo2 = _safe_num(a["Promotion_Rate"]), _safe_num(b["Promotion_Rate"])
    if abs(promo1 - promo2) >= 10:
        higher = p1 if promo1 > promo2 else p2
        reasons.append(
            f"{higher} was promoted more often ({_fmt(max(promo1, promo2))}% vs "
            f"{_fmt(min(promo1, promo2))}% of historical records), which may help explain "
            "part of the demand difference."
        )

    if not reasons:
        reasons.append(
            "The available data does not show one dominant driver; the difference is primarily "
            "visible in the observed demand, inventory and forecast-risk measures."
        )

    # Priority recommendation.
    risk_col = _col(summary, "Stockout Risk", "Risk Score")
    r1 = _safe_num(a.get(risk_col)) if risk_col else np.nan
    r2 = _safe_num(b.get(risk_col)) if risk_col else np.nan
    if pd.notna(r1) and pd.notna(r2) and r1 != r2:
        priority = p1 if r1 > r2 else p2
    elif pd.notna(days1) and pd.notna(days2):
        priority = p1 if _safe_num(days1) < _safe_num(days2) else p2
    else:
        priority = p1 if _safe_num(a.get("Current Inventory")) < _safe_num(b.get("Current Inventory")) else p2

    pr = summary[summary["Product ID"] == priority].iloc[0]
    order = pr.get("Recommended Order")
    recommendation = (
        f"Prioritize {priority} first. "
        f"The optimizer recommends approximately {_fmt(order, 0)} units for replenishment."
        if pd.notna(order)
        else f"Prioritize {priority} first based on its stronger inventory-pressure signals."
    )

    return {
        "type": "product_comparison",
        "headline": f"{p1} vs {p2} in {store_id}",
        "summary_table": _records(summary.drop(columns=[]), limit=10),
        "factor_table": _records(factors, limit=10),
        "recent_table": _records(recent_stats, limit=10),
        "reasons": reasons,
        "recommendation": recommendation,
        "charts": [
            _line_chart("Historical demand — last 120 observations", line_records, "Date", products),
            _bar_chart(
                "Inventory, forecast and replenishment comparison",
                bars,
                "Product",
                ["Current Inventory", "P50 Forecast", "P90 Forecast", "Recommended Order"],
            ),
        ],
    }


# ---------------------------------------------------------------------
# Risk query
# ---------------------------------------------------------------------
def get_top_risk_products(question: str = "") -> Dict[str, Any]:
    rec = _load_recommendations()
    if rec.empty:
        return {"error": "Inventory optimization output is not available."}

    risk_col = _col(rec, "Stockout Risk", "Risk Score")
    if risk_col:
        rec = rec.sort_values([risk_col, "Days Until Stockout"], ascending=[False, True])

    cols = [
        "Store ID", "Product ID", "Category", "Current Inventory",
        "Forecast P50 Demand", "Forecast P90 Demand", "Days Until Stockout",
        "Inventory Coverage Days", "Stockout Risk", "Risk Score", "Risk Level",
        "Recommended Order", "Projected Shortage", "Action Priority"
    ]
    top = rec.head(10).copy()

    chart_df = top.copy()
    chart_df["Series"] = chart_df["Store ID"].astype(str) + " / " + chart_df["Product ID"].astype(str)

    reasons = []
    if not top.empty:
        worst = top.iloc[0]
        reasons.append(
            f"{worst['Store ID']} / {worst['Product ID']} is the most urgent series because "
            f"its projected stockout is {_fmt(worst.get('Days Until Stockout'), 2)} days away "
            f"with {_fmt(worst.get('Current Inventory'), 0)} units currently available."
        )
        avg_risk = _safe_num(rec[risk_col].mean()) if risk_col else np.nan
        if pd.notna(avg_risk):
            reasons.append(
                f"The top-risk set has an average risk score of {_fmt(avg_risk, 1)}, "
                "indicating broad inventory pressure across the portfolio."
            )

    return {
        "type": "risk",
        "top_risks": _records(top, cols, 10),
        "reasons": reasons,
        "recommendation": "Start replenishment with the shortest stockout timelines, then use recommended order quantities to sequence the remaining high-risk series.",
        "charts": [
            _bar_chart(
                "Top stockout-risk series",
                chart_df[["Series", risk_col]].rename(columns={risk_col: "Risk"}).to_dict("records"),
                "Series",
                ["Risk"],
            ),
            _bar_chart(
                "Current inventory vs P50 demand",
                [
                    {
                        "Series": f"{r['Store ID']} / {r['Product ID']}",
                        "Current Inventory": r.get("Current Inventory"),
                        "P50 Demand": r.get("Forecast P50 Demand"),
                    }
                    for _, r in top.iterrows()
                ],
                "Series",
                ["Current Inventory", "P50 Demand"],
            ),
        ],
    }


# ---------------------------------------------------------------------
# Store analytics
# ---------------------------------------------------------------------
def get_store_analytics(question: str = "") -> Dict[str, Any]:
    df, rec = _load_sales(), _load_recommendations()

    s = df.groupby("Store ID", as_index=False).agg(
        Average_Daily_Demand=("Demand", "mean"),
        Total_Historical_Demand=("Demand", "sum"),
        Average_Inventory=("Inventory Level", "mean"),
        Average_Price=("Price", "mean"),
        Average_Discount=("Discount", "mean"),
    )

    if not rec.empty:
        e = rec.groupby("Store ID", as_index=False).agg(
            Current_Inventory=("Current Inventory", "sum"),
            Forecast_P50_Demand=("Forecast P50 Demand", "sum"),
            Forecast_P90_Demand=("Forecast P90 Demand", "sum"),
            Recommended_Order=("Recommended Order", "sum"),
            Projected_Shortage=("Projected Shortage", "sum"),
            High_Risk=("Risk Level", lambda x: (x.astype(str).str.lower() == "high").sum()),
        )
        s = s.merge(e, on="Store ID", how="left")

    s = s.sort_values("Average_Daily_Demand", ascending=False)

    reasons = []
    if not s.empty:
        highest_demand = s.iloc[0]
        reasons.append(
            f"{highest_demand['Store ID']} has the highest historical average daily demand "
            f"at {_fmt(highest_demand['Average_Daily_Demand'])} units."
        )
        if "Current_Inventory" in s.columns:
            coverage = s["Current_Inventory"] / s["Forecast_P50_Demand"].replace(0, np.nan) * 7
            s = s.assign(Forecast_Coverage_Days=coverage)
            tight = s.sort_values("Forecast_Coverage_Days").iloc[0]
            reasons.append(
                f"{tight['Store ID']} has the tightest forecast coverage at "
                f"{_fmt(tight['Forecast_Coverage_Days'], 2)} days."
            )

    return {
        "type": "store",
        "stores": _records(s, limit=20),
        "reasons": reasons,
        "recommendation": "Use demand together with forecast coverage and shortage exposure when prioritizing store-level replenishment.",
        "charts": [
            _bar_chart(
                "Average daily demand by store",
                s[["Store ID", "Average_Daily_Demand"]].to_dict("records"),
                "Store ID",
                ["Average_Daily_Demand"],
            ),
            _bar_chart(
                "Current inventory vs forecast demand by store",
                s[["Store ID", "Current_Inventory", "Forecast_P50_Demand"]].to_dict("records"),
                "Store ID",
                ["Current_Inventory", "Forecast_P50_Demand"],
            ),
        ],
    }


# ---------------------------------------------------------------------
# Product analytics
# ---------------------------------------------------------------------
def get_product_analytics(question: str = "") -> Dict[str, Any]:
    df, rec = _load_sales(), _load_recommendations()

    p = df.groupby("Product ID", as_index=False).agg(
        Average_Daily_Demand=("Demand", "mean"),
        Total_Historical_Demand=("Demand", "sum"),
        Average_Inventory=("Inventory Level", "mean"),
    )

    if not rec.empty:
        e = rec.groupby("Product ID", as_index=False).agg(
            Current_Inventory=("Current Inventory", "sum"),
            Forecast_P50_Demand=("Forecast P50 Demand", "sum"),
            Forecast_P90_Demand=("Forecast P90 Demand", "sum"),
            Recommended_Order=("Recommended Order", "sum"),
            Projected_Shortage=("Projected Shortage", "sum"),
        )
        p = p.merge(e, on="Product ID", how="left")

    p["Forecast_Coverage_Days"] = p["Current_Inventory"] / p["Forecast_P50_Demand"].replace(0, np.nan) * 7
    risk_rank = p.sort_values("Projected_Shortage", ascending=False).head(10)

    reasons = []
    if not p.empty:
        high_demand = p.sort_values("Average_Daily_Demand", ascending=False).iloc[0]
        reasons.append(
            f"{high_demand['Product ID']} has the highest historical average demand at "
            f"{_fmt(high_demand['Average_Daily_Demand'])} units/day."
        )
        tight = p.sort_values("Forecast_Coverage_Days").iloc[0]
        reasons.append(
            f"{tight['Product ID']} has the tightest forecast coverage at "
            f"{_fmt(tight['Forecast_Coverage_Days'], 2)} days."
        )

    return {
        "type": "product",
        "products": _records(p.sort_values("Average_Daily_Demand", ascending=False), limit=20),
        "reasons": reasons,
        "recommendation": "Products with high forecast demand and low coverage should receive replenishment attention first.",
        "charts": [
            _bar_chart(
                "Top products by historical demand",
                p.sort_values("Average_Daily_Demand", ascending=False).head(10)[
                    ["Product ID", "Average_Daily_Demand"]
                ].to_dict("records"),
                "Product ID",
                ["Average_Daily_Demand"],
            ),
            _bar_chart(
                "Top projected shortage by product",
                risk_rank[["Product ID", "Projected_Shortage"]].to_dict("records"),
                "Product ID",
                ["Projected_Shortage"],
            ),
        ],
    }


# ---------------------------------------------------------------------
# Dataset / driver analytics
# ---------------------------------------------------------------------
def analyze_dataset(question: str = "") -> Dict[str, Any]:
    df = _load_sales()
    q = question.lower()
    evidence: Dict[str, Any] = {
        "rows": len(df),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "average_demand": round(float(df["Demand"].mean()), 2),
    }
    charts = []
    reasons = []

    if "promotion" in q:
        x = df.assign(
            Promotion=df["Promotion"].astype(str).str.title()
        ).groupby("Promotion", as_index=False)["Demand"].mean().rename(columns={"Demand": "Average_Demand"})
        evidence["promotion"] = _records(x.round(2))
        charts.append(_bar_chart("Average demand: promotion vs no promotion", x.to_dict("records"), "Promotion", ["Average_Demand"]))
        if len(x) >= 2:
            reasons.append(
                f"Average demand is {_fmt(x['Average_Demand'].max())} in the higher-demand promotion state "
                f"versus {_fmt(x['Average_Demand'].min())} in the other state."
            )

    if "discount" in q:
        x = df.groupby("Discount", as_index=False)["Demand"].mean().rename(columns={"Demand": "Average_Demand"})
        evidence["discount"] = _records(x.round(2))
        charts.append(_line_chart("Average demand by discount level", x.to_dict("records"), "Discount", ["Average_Demand"]))

    if "month" in q or "season" in q:
        x = (
            df.assign(Month=df["Date"].dt.month)
            .groupby("Month", as_index=False)["Demand"].mean()
            .rename(columns={"Demand": "Average_Demand"})
        )
        x["Month"] = x["Month"].map(lambda m: pd.Timestamp(2026, int(m), 1).strftime("%b"))
        evidence["monthly"] = _records(x.round(2))
        charts.append(_line_chart("Average demand by month", x.to_dict("records"), "Month", ["Average_Demand"]))

    # Default analytical view if the query did not specify a driver.
    if not charts:
        daily = df.groupby("Date", as_index=False)["Demand"].sum().tail(120)
        daily["Date"] = daily["Date"].dt.strftime("%Y-%m-%d")
        evidence["daily_demand"] = _records(daily, limit=120)
        charts.append(_line_chart("Daily total demand — recent 120 days", daily.to_dict("records"), "Date", ["Demand"]))
        reasons.append(
            f"The dataset contains {len(df):,} records with average demand of "
            f"{_fmt(df['Demand'].mean())} units per record."
        )

    evidence["reasons"] = reasons
    evidence["charts"] = charts
    return evidence


# ---------------------------------------------------------------------
# What-if scenario
# ---------------------------------------------------------------------
def run_what_if_scenario(
    question: str = "",
    demand_change_pct: float = 20,
    store_id: str | None = None,
    product_id: str | None = None,
) -> Dict[str, Any]:
    rec = _load_recommendations()
    if rec.empty:
        return {"error": "Inventory optimization output is not available."}

    if store_id:
        rec = rec[rec["Store ID"].astype(str).str.upper() == store_id.upper()]
    if product_id:
        rec = rec[rec["Product ID"].astype(str).str.upper() == product_id.upper()]

    p50 = _col(rec, "Forecast P50 Demand", "P50 Demand")
    p90 = _col(rec, "Forecast P90 Demand", "P90 Demand")
    if not p50 or not p90:
        return {"error": "Forecast demand columns are not available."}

    inventory = _safe_num(rec["Current Inventory"].sum())
    baseline_p50 = _safe_num(rec[p50].sum())
    baseline_p90 = _safe_num(rec[p90].sum())

    multiplier = 1 + demand_change_pct / 100
    scenario_p50 = baseline_p50 * multiplier
    scenario_p90 = baseline_p90 * multiplier

    baseline_shortage = max(baseline_p90 - inventory, 0)
    scenario_shortage = max(scenario_p90 - inventory, 0)
    baseline_order = max(baseline_p90 - inventory, 0)
    scenario_order = max(scenario_p90 - inventory, 0)

    return {
        "type": "scenario",
        "demand_change_pct": demand_change_pct,
        "baseline_p50": round(baseline_p50, 2),
        "scenario_p50": round(scenario_p50, 2),
        "baseline_p90": round(baseline_p90, 2),
        "scenario_p90": round(scenario_p90, 2),
        "current_inventory": round(inventory, 2),
        "baseline_shortage": round(baseline_shortage, 2),
        "scenario_shortage": round(scenario_shortage, 2),
        "baseline_order": round(baseline_order, 2),
        "scenario_order": round(scenario_order, 2),
        "reasons": [
            f"A {demand_change_pct:+.0f}% demand change moves P50 demand from "
            f"{_fmt(baseline_p50, 0)} to {_fmt(scenario_p50, 0)} units.",
            f"Projected P90 shortage changes from {_fmt(baseline_shortage, 0)} "
            f"to {_fmt(scenario_shortage, 0)} units."
        ],
        "recommendation": (
            f"Under this stress scenario, plan for approximately {_fmt(scenario_order, 0)} "
            "units of replenishment exposure rather than the baseline requirement."
        ),
        "charts": [
            _bar_chart(
                "Baseline vs demand scenario",
                [
                    {"Metric": "P50 Demand", "Baseline": baseline_p50, "Scenario": scenario_p50},
                    {"Metric": "P90 Demand", "Baseline": baseline_p90, "Scenario": scenario_p90},
                    {"Metric": "Projected Shortage", "Baseline": baseline_shortage, "Scenario": scenario_shortage},
                    {"Metric": "Recommended Order", "Baseline": baseline_order, "Scenario": scenario_order},
                ],
                "Metric",
                ["Baseline", "Scenario"],
            )
        ],
    }


# ---------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------
def choose_analysis(question: str) -> str:
    products = _extract_product_ids(question)
    stores = _extract_store_ids(question)
    q = question.lower()

    if len(products) >= 2 and len(stores) >= 1 and any(
        w in q for w in ["compare", "comparison", "versus", " vs ", "difference", "better", "between"]
    ):
        return "compare_products"

    if any(w in q for w in ["stockout", "risk", "run out", "shortage", "replenish", "replenishment"]):
        return "risk"

    if any(w in q for w in ["what if", "scenario", "increase demand", "decrease demand"]):
        return "scenario"

    if any(w in q for w in ["promotion", "discount", "month", "season", "historical"]):
        return "dataset"

    if "store" in q:
        return "store"

    if "product" in q or "sku" in q:
        return "product"

    return "dataset"


# ---------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------
def ask_agent(question: str) -> Dict[str, Any]:
    question = question.strip()
    if not question:
        return {
            "answer": "Please enter a business question.",
            "tools_used": [],
            "evidence": {},
            "visuals": [],
        }

    kind = choose_analysis(question)
    products = _extract_product_ids(question)
    stores = _extract_store_ids(question)

    if kind == "compare_products":
        e = compare_products(question, stores[0], products)
        if "error" in e:
            return {"answer": e["error"], "tools_used": ["compare_products"], "evidence": e, "visuals": []}

        p1, p2 = e["headline"].split(" vs ", 1)[0], e["headline"].split(" vs ", 1)[1].split(" in ")[0]
        reasons = " ".join(e.get("reasons", []))
        answer = (
            f"**{e['headline']}**\n\n"
            f"**Conclusion:** {reasons}\n\n"
            f"**Recommended action:** {e['recommendation']}"
        )
        return {
            "answer": answer,
            "tools_used": ["compare_products"],
            "evidence": e,
            "visuals": e["charts"],
        }

    if kind == "risk":
        e = get_top_risk_products(question)
        if "error" in e:
            return {"answer": e["error"], "tools_used": ["get_top_risk_products"], "evidence": e, "visuals": []}
        answer = (
            "**Stockout risk analysis**\n\n"
            + " ".join(e.get("reasons", []))
            + "\n\n**Recommended action:** "
            + e.get("recommendation", "")
        )
        return {"answer": answer, "tools_used": ["get_top_risk_products"], "evidence": e, "visuals": e["charts"]}

    if kind == "scenario":
        e = run_what_if_scenario(
            question,
            _extract_percent(question),
            stores[0] if stores else None,
            products[0] if products else None,
        )
        if "error" in e:
            return {"answer": e["error"], "tools_used": ["run_what_if_scenario"], "evidence": e, "visuals": []}
        answer = (
            f"**Scenario analysis:** demand changes by {e['demand_change_pct']:+.0f}%.\n\n"
            f"P50 demand changes from **{_fmt(e['baseline_p50'], 0)}** to "
            f"**{_fmt(e['scenario_p50'], 0)}** units, while projected shortage changes from "
            f"**{_fmt(e['baseline_shortage'], 0)}** to **{_fmt(e['scenario_shortage'], 0)}** units.\n\n"
            f"**Recommended action:** {e['recommendation']}"
        )
        return {"answer": answer, "tools_used": ["run_what_if_scenario"], "evidence": e, "visuals": e["charts"]}

    if kind == "store":
        e = get_store_analytics(question)
        answer = (
            "**Store analysis**\n\n"
            + " ".join(e.get("reasons", []))
            + "\n\n**Recommended action:** "
            + e.get("recommendation", "")
        )
        return {"answer": answer, "tools_used": ["get_store_analytics"], "evidence": e, "visuals": e["charts"]}

    if kind == "product":
        e = get_product_analytics(question)
        answer = (
            "**Product analysis**\n\n"
            + " ".join(e.get("reasons", []))
            + "\n\n**Recommended action:** "
            + e.get("recommendation", "")
        )
        return {"answer": answer, "tools_used": ["get_product_analytics"], "evidence": e, "visuals": e["charts"]}

    e = analyze_dataset(question)
    answer = (
        "**Dataset analysis**\n\n"
        + " ".join(e.get("reasons", []))
        + "\n\nThe charts and validation evidence below are generated directly from the project CSV."
    )
    return {"answer": answer, "tools_used": ["analyze_dataset"], "evidence": e, "visuals": e["charts"]}

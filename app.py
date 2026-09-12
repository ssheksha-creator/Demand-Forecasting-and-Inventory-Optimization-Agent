
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from agent import ask_agent

from analytics import (
    load_data,
    get_business_summary,
    get_store_analytics,
    get_product_analytics,
    get_category_analytics,
    get_risk_distribution,
    get_top_risk_products,
    get_model_performance,
    get_monthly_demand,
    get_promotion_analytics,
    get_discount_analytics,
    get_weather_analytics,
    get_epidemic_analytics,
    get_region_analytics,
    get_seasonality_analytics,
)

st.set_page_config(
    page_title="AI Inventory Copilot",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------
st.markdown("""
<style>
.stApp { background:#f5f7fb; color:#172033; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#101827; border-right:1px solid #202b3e; }
[data-testid="stSidebar"] * { color:#eef2f7; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1450px; }
.brand { padding:8px 4px 22px; }
.brand-icon { width:42px;height:42px;border-radius:12px;background:#fff;color:#101827;
display:flex;align-items:center;justify-content:center;font-size:21px;margin-bottom:12px; }
.brand-title { font-size:20px;font-weight:800;color:#fff; }
.brand-sub { font-size:12px;color:#91a0b5;margin-top:5px;line-height:1.5; }
.side-label { font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#8190a6;
font-weight:800;margin:20px 0 8px 4px; }
.side-note { margin-top:20px;padding:12px;border:1px solid #26334a;border-radius:12px;
background:#151f31;color:#9eabc0;font-size:11px;line-height:1.55; }
.hero { background:#fff;border:1px solid #e6eaf0;border-radius:18px;padding:22px 25px;margin-bottom:18px; }
.eyebrow { display:inline-block;background:#eef5ff;color:#2f6fed;border-radius:999px;
padding:5px 10px;font-size:10px;font-weight:800;letter-spacing:.07em;text-transform:uppercase; }
.hero h1 { margin:9px 0 4px;font-size:30px;letter-spacing:-.02em; }
.hero p { color:#718096;margin:0;font-size:14px; }
.kpi { background:#fff;border:1px solid #e5eaf1;border-radius:15px;padding:17px 18px;min-height:112px; }
.kpi-label { color:#7b8798;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.07em; }
.kpi-value { color:#172033;font-size:26px;font-weight:800;margin-top:9px; }
.kpi-sub { color:#8792a3;font-size:11px;margin-top:4px; }
.section { font-size:17px;font-weight:800;margin:25px 0 12px; }
.card { background:#fff;border:1px solid #e5eaf1;border-radius:15px;padding:16px; }
.risk-high { color:#d33b40;background:#fff0f0;border-radius:999px;padding:4px 8px;font-weight:800;font-size:11px; }
.risk-medium { color:#ad7200;background:#fff7df;border-radius:999px;padding:4px 8px;font-weight:800;font-size:11px; }
.risk-low { color:#168052;background:#eaf8f1;border-radius:999px;padding:4px 8px;font-weight:800;font-size:11px; }
.small-muted { color:#7b8798;font-size:12px; }
[data-testid="stMetric"] { background:#fff;border:1px solid #e5eaf1;padding:14px;border-radius:14px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Data -----------------------------
@st.cache_data
def data():
    return load_data()

@st.cache_data
def recommendations():
    p = ROOT / "outputs" / "recommendations_v2.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def forecasts():
    p = ROOT / "outputs" / "forecasts_v2.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

df = data()
rec = recommendations()
fc = forecasts()

# Compatibility aliases: analytics/inventory outputs use descriptive snake-case
# names while some dashboard views use shorter display names. Keep both available.
if not rec.empty:
    for src, dst in {
        "Forecast P50 Demand": "P50 Demand",
        "Forecast P90 Demand": "P90 Demand",
        "Forecast P10 Demand": "P10 Demand",
        "Stockout Risk": "Risk Score",
    }.items():
        if src in rec.columns and dst not in rec.columns:
            rec[dst] = rec[src]

summary = get_business_summary(df)

# ----------------------------- Helpers -----------------------------
def money(x): return f"{x:,.0f}"
def num(x): return f"{x:,.0f}"
def pct(x): return f"{x:.1f}%"

def hero(title, subtitle, eyebrow="AI INVENTORY COPILOT"):
    st.markdown(
        f'<div class="hero"><span class="eyebrow">{eyebrow}</span>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def kpis(items):
    cols = st.columns(len(items))
    for c, (label, value, sub) in zip(cols, items):
        with c:
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

def chart(fig, height=350):
    fig.update_layout(
        height=height, margin=dict(l=8,r=8,t=42,b=8),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Arial", color="#344054"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)

def get_metric(metrics, model="LightGBM V2", metric="MAPE"):
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return None
    x = metrics[metrics["Model"].astype(str).str.contains(model, case=False, na=False)]
    if x.empty or metric not in x.columns: return None
    return float(x[metric].mean())

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">📦</div>
      <div class="brand-title">AI Inventory Copilot</div>
      <div class="brand-sub">Demand forecasting, inventory risk and replenishment intelligence.</div>
    </div>
    <div class="side-label">Workspace</div>
    """, unsafe_allow_html=True)

    pages = [
        "🏠 Dashboard", "📈 Demand Forecast", "⚠️ Inventory Risk",
        "📦 Replenishment", "🔮 What-If Simulator", "🤖 AI Agent",
        "🏪 Store Analytics", "🛍️ Product Analytics", "📊 Model Performance"
    ]
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown(
        f'<div class="side-note"><b>Data foundation</b><br>'
        f'{len(df):,} historical records · {df["Store ID"].nunique()} stores · '
        f'{df["Product ID"].nunique()} products<br><br>'
        f'<b>Decision flow</b><br>Predict → Detect → Optimize → Explain → Simulate → Act</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- Dashboard -----------------------------
if page == "🏠 Dashboard":
    hero("Inventory command center",
         "See demand, inventory exposure and the actions that need attention first.")

    kpis([
        ("Total inventory", num(summary["total_inventory"]), "Units currently on hand"),
        ("7-day P50 demand", num(summary["forecast_demand_p50"]), "Expected demand"),
        ("High-risk series", num(summary["high_risk"]), "Store × product combinations"),
        ("Forecast MAPE", pct(summary.get("forecast_mape", 15.0)), "Lower is better"),
    ])

    section("Attention required")
    if not rec.empty:
        sort_columns = [c for c in ["Risk Score", "Recommended Order"] if c in rec.columns]
        show = rec.sort_values(sort_columns, ascending=False).head(10).copy() if sort_columns else rec.head(10).copy()
        cols = [c for c in ["Store ID","Product ID","Category","Current Inventory",
                            "Risk Level","Days Until Stockout","Recommended Order","Action Priority"] if c in show]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Run Phase 3 inventory optimization to populate recommendations.")

    c1, c2 = st.columns(2)
    with c1:
        section("Historical demand trend")
        daily = df.groupby("Date", as_index=False)["Demand"].sum()
        fig = px.line(daily, x="Date", y="Demand", title="Daily total demand")
        chart(fig, 320)
    with c2:
        section("Store demand comparison")
        s = df.groupby("Store ID", as_index=False)["Demand"].mean().sort_values("Demand", ascending=False)
        fig = px.bar(s, x="Store ID", y="Demand", title="Average daily demand by store")
        chart(fig, 320)

    c1, c2 = st.columns(2)
    with c1:
        section("Inventory vs forecast")
        if not rec.empty:
            x = rec.groupby("Store ID", as_index=False).agg(
                Current=("Current Inventory","sum"),
                Forecast=("Forecast P50 Demand","sum")
            )
            fig = go.Figure()
            fig.add_bar(x=x["Store ID"], y=x["Current"], name="Current inventory")
            fig.add_bar(x=x["Store ID"], y=x["Forecast"], name="7-day P50 demand")
            fig.update_layout(barmode="group", title="Store-level coverage pressure")
            chart(fig, 330)
    with c2:
        section("Risk distribution")
        r = get_risk_distribution(df)
        fig = px.bar(r, x="Risk Level", y="Count", title="Store × product risk")
        chart(fig, 330)

# ----------------------------- Demand Forecast -----------------------------
elif page == "📈 Demand Forecast":
    hero("Demand forecast", "Explore historical behavior and the next 7 days of model-driven demand.")

    stores = sorted(df["Store ID"].unique())
    products = sorted(df["Product ID"].unique())
    c1,c2,c3 = st.columns(3)
    store = c1.selectbox("Store", stores)
    product = c2.selectbox("Product", products)
    horizon = c3.selectbox("View", ["7 days", "30 days"])

    hist = df[(df["Store ID"]==store)&(df["Product ID"]==product)].sort_values("Date")
    if not fc.empty:
        fut = fc[(fc["Store ID"]==store)&(fc["Product ID"]==product)].sort_values("Date")
    else: fut = pd.DataFrame()

    section("Historical → forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"].tail(120), y=hist["Demand"].tail(120),
                             name="Historical demand", mode="lines"))
    if not fut.empty:
        fview = fut.head(7 if horizon=="7 days" else 30)
        mid = "Predicted Demand" if "Predicted Demand" in fview else "Forecast Demand"
        p10 = "P10 Demand" if "P10 Demand" in fview else None
        p90 = "P90 Demand" if "P90 Demand" in fview else None
        if p10 and p90:
            fig.add_trace(go.Scatter(x=fview["Date"], y=fview[p90], name="P90", mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=fview["Date"], y=fview[p10], name="P10",
                                     fill="tonexty", mode="lines", line=dict(width=0)))
        if mid in fview:
            fig.add_trace(go.Scatter(x=fview["Date"], y=fview[mid], name="P50 forecast", mode="lines+markers"))
    chart(fig, 430)

    c1,c2 = st.columns(2)
    with c1:
        section("Store comparison")
        s = df.groupby("Store ID", as_index=False)["Demand"].mean()
        chart(px.bar(s, x="Store ID", y="Demand", title="Average demand"))
    with c2:
        section("Monthly seasonality")
        m = get_monthly_demand(df)
        chart(px.line(m, x="Month_Name", y="Average_Demand", markers=True, title="Average demand by month"))

# ----------------------------- Inventory Risk -----------------------------
elif page == "⚠️ Inventory Risk":
    hero("Inventory risk", "Prioritize the store-product combinations most likely to run out.")
    kpis([
        ("High risk", num(summary["high_risk"]), "Immediate attention"),
        ("Medium risk", num(summary["medium_risk"]), "Watch closely"),
        ("Low risk", num(summary["low_risk"]), "Healthy coverage"),
        ("Avg stockout risk", pct(summary["average_stockout_risk"]), "Model-derived risk score"),
    ])
    c1,c2 = st.columns([1.3,1])
    with c1:
        section("Top risk products")
        if not rec.empty:
            sort_columns = [c for c in ["Risk Score", "Projected Shortage"] if c in rec.columns]
            show = rec.sort_values(sort_columns, ascending=False).head(15) if sort_columns else rec.head(15)
            chart(px.bar(show, x="Stockout Risk", y="Product ID", color="Risk Level",
                         hover_data=["Store ID","Current Inventory","Recommended Order"],
                         orientation="h", title="Highest-risk store-product series"), 450)
    with c2:
        section("Risk distribution")
        r = get_risk_distribution(df)
        chart(px.pie(r, names="Risk Level", values="Count", hole=.58, title="Risk mix"), 450)
    section("Risk register")
    if not rec.empty:
        cols = [c for c in ["Store ID","Product ID","Category","Region","Current Inventory",
                            "P50 Demand","P90 Demand","Safety Stock","Reorder Point",
                            "Days Until Stockout","Risk Score","Risk Level","Action Priority"] if c in rec]
        risk_sorted = rec.sort_values("Risk Score", ascending=False) if "Risk Score" in rec.columns else rec
        st.dataframe(risk_sorted[cols], use_container_width=True, hide_index=True)

# ----------------------------- Replenishment -----------------------------
elif page == "📦 Replenishment":
    hero("Replenishment planner", "Turn forecasted demand and risk into prioritized order quantities.")
    if rec.empty:
        st.warning("No recommendations found. Generate outputs/recommendations_v2.csv first.")
    else:
        kpis([
            ("Recommended units", num(rec["Recommended Order"].sum()), "Across all series"),
            ("Projected shortage", num(rec["Projected Shortage"].sum()), "Before replenishment"),
            ("Urgent actions", num((rec["Action Priority"]=="Urgent Replenishment").sum()), "Highest priority"),
            ("Avg coverage", f'{rec["Inventory Coverage Days"].mean():.1f} d', "Current inventory coverage"),
        ])
        c1,c2 = st.columns(2)
        with c1:
            x = rec.groupby("Store ID", as_index=False)["Recommended Order"].sum()
            chart(px.bar(x, x="Store ID", y="Recommended Order", title="Recommended order by store"), 350)
        with c2:
            x = rec.groupby("Product ID", as_index=False)["Recommended Order"].sum().sort_values("Recommended Order", ascending=False).head(10)
            chart(px.bar(x, x="Recommended Order", y="Product ID", orientation="h",
                         title="Top replenishment quantities"), 350)
        section("Priority orders")
        cols = [c for c in ["Store ID","Product ID","Category","Current Inventory",
                            "Reorder Point","Target Inventory","Recommended Order",
                            "Projected Shortage","Risk Level","Action Priority"] if c in rec]
        sort_columns = [c for c in ["Risk Score", "Recommended Order"] if c in rec.columns]
        risk_sorted = rec.sort_values(sort_columns, ascending=False) if sort_columns else rec
        st.dataframe(risk_sorted[cols],
                     use_container_width=True, hide_index=True)

# ----------------------------- What-if -----------------------------
elif page == "🔮 What-If Simulator":
    hero("What-if simulator", "Stress-test demand and replenishment decisions before changing the plan.")
    c1,c2,c3 = st.columns(3)
    demand_change = c1.slider("Demand change", -30, 50, 20, 5)
    lead_time = c2.slider("Lead time (days)", 1, 14, 3)
    discount = c3.slider("Discount (%)", 0, 30, 10, 5)

    if rec.empty:
        st.warning("Recommendations are required for the simulator.")
    else:
        base = rec.groupby("Store ID", as_index=False).agg(
            Inventory=("Current Inventory","sum"),
            P50=("Forecast P50 Demand","sum"),
            P90=("Forecast P90 Demand","sum"),
            Order=("Recommended Order","sum"),
            Shortage=("Projected Shortage","sum"),
        )
        scenario = base.copy()
        uplift = 1 + demand_change/100
        # Conservative scenario: scale demand and shortage; keep inventory fixed.
        scenario["P50"] *= uplift
        scenario["P90"] *= uplift
        scenario["Shortage"] = np.maximum(scenario["P90"] - scenario["Inventory"], 0)
        scenario["Order"] = np.maximum(scenario["P90"] - scenario["Inventory"], 0).round()
        scenario["Lead Time"] = lead_time

        k1,k2,k3 = st.columns(3)
        k1.metric("Scenario P50 demand", num(scenario["P50"].sum()),
                  f"{demand_change:+d}% vs baseline")
        k2.metric("Scenario shortage", num(scenario["Shortage"].sum()),
                  f"{scenario['Shortage'].sum()-base['Shortage'].sum():+,.0f} units")
        k3.metric("Scenario order", num(scenario["Order"].sum()),
                  f"{scenario['Order'].sum()-base['Order'].sum():+,.0f} units")

        section("Baseline vs scenario")
        compare = pd.DataFrame({
            "Metric":["P50 demand","P90 demand","Recommended order","Projected shortage"],
            "Baseline":[base.P50.sum(),base.P90.sum(),base.Order.sum(),base.Shortage.sum()],
            "Scenario":[scenario.P50.sum(),scenario.P90.sum(),scenario.Order.sum(),scenario.Shortage.sum()]
        })
        long = compare.melt(id_vars="Metric", var_name="Plan", value_name="Units")
        chart(px.bar(long, x="Metric", y="Units", color="Plan", barmode="group",
                     title="Decision impact"), 390)

        st.info(
            f"**AI interpretation:** A {demand_change:+d}% demand shock raises the modeled demand exposure. "
            f"At a {lead_time}-day lead time, prioritize high-risk series first and use the scenario order quantity "
            f"as a stress-test—not as a replacement for the trained forecast."
        )

# ----------------------------- AI Agent -----------------------------
elif page == "🤖 AI Agent":
    hero("AI decision analyst", "Ask a business question and get the answer, query-specific visuals, validation evidence and recommended action in one workspace.")

    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    suggestions = [
        "Compare P0018 and P0016 in S002",
        "Compare P0009 and P0017 in S001",
        "Which products are highest risk?",
        "Which store has the most inventory pressure?",
        "What happens if demand increases by 20%?",
        "How does promotion affect demand?",
    ]

    st.markdown("<div class='card'><b>Ask the copilot</b><br><span class='small-muted'>Try a comparison such as: Compare P0018 and P0016 in S002</span></div>", unsafe_allow_html=True)
    q = st.text_input("Query", placeholder="Compare P0018 and P0016 in S002", label_visibility="collapsed")

    if st.button("Analyze query", type="primary", use_container_width=False) and q.strip():
        result = ask_agent(q)
        st.session_state.agent_history.append((q, result))

    # Render latest result as a complete query workspace.
    if st.session_state.agent_history:
        query, result = st.session_state.agent_history[-1]
        st.markdown(f"### 🔎 {query}")

        st.markdown("#### AI explanation")
        st.markdown(result.get("answer", "No answer returned."))

        evidence = result.get("evidence", {}) or {}
        visuals = result.get("visuals", []) or []

        # Query-specific visualizations.
        if visuals:
            st.markdown("#### 📊 Query-specific charts")
            for visual in visuals:
                vtype = visual.get("type")
                if vtype == "line":
                    rows = visual.get("data", [])
                    xkey = visual.get("x", "Date")
                    series = visual.get("series", [])
                    fig = go.Figure()
                    for product_id in series:
                        ys = [r.get(product_id) for r in rows]
                        xs = [r.get(xkey) for r in rows]
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=product_id))
                    fig.update_layout(title=visual.get("title", "Trend comparison"), xaxis_title=xkey, yaxis_title="Demand")
                    chart(fig, 400)
                elif vtype == "bar":
                    rows = pd.DataFrame(visual.get("data", []))
                    if not rows.empty:
                        xkey = visual.get("x", "Product")
                        series = [c for c in visual.get("series", []) if c in rows.columns]
                        fig = go.Figure()
                        for col in series:
                            fig.add_bar(x=rows[xkey], y=rows[col], name=col)
                        fig.update_layout(title=visual.get("title", "Comparison"), barmode="group", xaxis_title=xkey, yaxis_title="Units")
                        chart(fig, 400)

        # Validation/evidence tables.
        st.markdown("#### 📋 Validation & evidence")
        table_keys = [
            ("summary_table", "Decision comparison"),
            ("factor_table", "Historical driver evidence"),
            ("recent_table", "Recent 30-observation evidence"),
            ("top_risks", "Risk validation table"),
            ("stores", "Store validation table"),
            ("products", "Product validation table"),
        ]
        rendered = False
        for key, title in table_keys:
            rows = evidence.get(key)
            if isinstance(rows, list) and rows:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                rendered = True
        if not rendered:
            scalar = {k: v for k, v in evidence.items() if not isinstance(v, (dict, list))}
            if scalar:
                st.dataframe(pd.DataFrame([scalar]), use_container_width=True, hide_index=True)
            else:
                st.info("No tabular validation evidence was returned for this query.")

        if evidence.get("reasons"):
            st.markdown("#### 🔎 Why the result differs")
            for reason in evidence["reasons"]:
                st.markdown(f"- {reason}")

        if evidence.get("recommendation"):
            st.markdown("#### 🎯 Recommended action")
            st.success(evidence["recommendation"])

        st.caption("Numbers are generated from the project CSV, trained forecast outputs and inventory optimization outputs. The AI explanation layer does not invent forecast values.")

    st.markdown("#### Suggested questions")
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion}"):
            result = ask_agent(suggestion)
            st.session_state.agent_history.append((suggestion, result))
            st.rerun()

# ----------------------------- Store Analytics -----------------------------
elif page == "🏪 Store Analytics":
    hero("Store analytics", "Compare demand, inventory exposure and replenishment pressure across stores.")
    s = get_store_analytics(df)
    st.dataframe(s, use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1:
        chart(px.bar(s, x="Store ID", y="Average_Daily_Demand", title="Average demand by store"))
    with c2:
        # analytics.py returns the canonical column name "High_Risk".
        # Keep a fallback for older output schemas.
        risk_col = next((c for c in ["High_Risk", "High Risk", "High-Risk"] if c in s.columns), None)
        if risk_col:
            chart(
                px.bar(
                    s,
                    x="Store ID",
                    y=risk_col,
                    title="High-risk series by store",
                    labels={risk_col: "High-risk series"},
                )
            )
        else:
            st.info("Risk detail is available in Inventory Risk.")

# ----------------------------- Product Analytics -----------------------------
elif page == "🛍️ Product Analytics":
    hero("Product analytics", "Identify demand leaders, weak movers and products creating inventory pressure.")
    p = get_product_analytics(df)
    if "Average Demand" not in p.columns and "Average_Daily_Demand" in p.columns:
        p["Average Demand"] = p["Average_Daily_Demand"]
    st.dataframe(p, use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1:
        if "Average Demand" in p:
            x=p.sort_values("Average Demand",ascending=False).head(10)
            chart(px.bar(x,x="Average Demand",y="Product ID",orientation="h",title="Top products by demand"))
    with c2:
        if not rec.empty:
            x=rec.groupby("Product ID",as_index=False)["Recommended Order"].sum().sort_values("Recommended Order",ascending=False).head(10)
            chart(px.bar(x,x="Recommended Order",y="Product ID",orientation="h",title="Products needing replenishment"))

# ----------------------------- Model Performance -----------------------------
elif page == "📊 Model Performance":
    hero("Model performance", "Validate forecast quality and high-demand detection using chronological validation.")

    # ---------------------------------------------------------
    # Primary forecasting metrics: MAE / RMSE / MAPE
    # ---------------------------------------------------------
    rv = ROOT / "outputs" / "rolling_validation_v2.csv"
    val = pd.read_csv(rv) if rv.exists() else pd.DataFrame()

    section("Forecasting metrics")
    if not val.empty and {"Model", "MAE", "RMSE", "MAPE"}.issubset(val.columns):
        # Mean across the four chronological validation folds.
        agg = (
            val.groupby("Model", as_index=False)[["MAE", "RMSE", "MAPE"]]
            .mean()
            .sort_values("MAE")
        )

        model_row = agg[agg["Model"].astype(str).str.contains("LightGBM", case=False, na=False)]
        if model_row.empty:
            model_row = agg.iloc[[0]]

        r = model_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{float(r['MAE']):.2f}")
        c2.metric("RMSE", f"{float(r['RMSE']):.2f}")
        c3.metric("MAPE", f"{float(r['MAPE']):.2f}%")

        st.caption("Primary metrics are averaged across the 4 chronological validation folds. Lower is better.")
        st.dataframe(agg.round(3), use_container_width=True, hide_index=True)

        chart(
            px.bar(
                agg,
                x="Model",
                y="MAE",
                title="Average MAE across validation folds — lower is better",
            ),
            360,
        )
    else:
        perf = get_model_performance()
        if isinstance(perf, dict):
            metrics = perf.get("metrics")
            if isinstance(metrics, pd.DataFrame):
                st.dataframe(metrics, use_container_width=True, hide_index=True)
            else:
                st.json(perf)

    # ---------------------------------------------------------
    # Chronological validation evidence
    # ---------------------------------------------------------
    if not val.empty:
        section("4-fold chronological validation")
        st.dataframe(val.round(3), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # Actual vs predicted + classification diagnostics
    # ---------------------------------------------------------
    vp = ROOT / "outputs" / "validation_predictions_v2.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        actual = next((c for c in ["Actual Demand", "Demand"] if c in v.columns), None)
        pred = next((c for c in ["Predicted Demand", "Prediction"] if c in v.columns), None)

        if actual and pred:
            eval_df = v[[actual, pred]].dropna().copy()

            section("Actual vs predicted")
            sample = eval_df.tail(500).copy()
            fig = px.scatter(
                sample,
                x=actual,
                y=pred,
                title="Validation predictions",
                labels={actual: "Actual demand", pred: "Predicted demand"},
            )
            fig.add_shape(
                type="line",
                x0=float(sample[actual].min()),
                x1=float(sample[actual].max()),
                y0=float(sample[actual].min()),
                y1=float(sample[actual].max()),
            )
            chart(fig, 420)

            # -------------------------------------------------
            # Classification diagnostics.
            # Demand forecasting itself is regression. To give
            # Precision / Recall / F1 / Accuracy a valid business
            # interpretation, classify observations as Normal vs
            # High Demand using the actual validation-set median.
            # -------------------------------------------------
            section("Classification diagnostics — high-demand detection")

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
            )

            if len(eval_df) >= 2:
                threshold = float(eval_df[actual].median())

                y_true = (eval_df[actual] >= threshold).astype(int)
                y_pred = (eval_df[pred] >= threshold).astype(int)

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                # Six visible KPI cards including the confusion-matrix counts.
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Accuracy", f"{accuracy:.1%}")
                k2.metric("Precision", f"{precision:.1%}")
                k3.metric("Recall", f"{recall:.1%}")
                k4.metric("F1 Score", f"{f1:.1%}")

                tn, fp, fn, tp = cm.ravel()

                k5, k6, k7, k8 = st.columns(4)
                k5.metric("True Negatives", f"{tn:,}")
                k6.metric("False Positives", f"{fp:,}")
                k7.metric("False Negatives", f"{fn:,}")
                k8.metric("True Positives", f"{tp:,}")

                st.caption(
                    f"High-demand threshold: {threshold:.2f} units. "
                    "Accuracy = overall correct classifications; Precision = correctness "
                    "of high-demand alerts; Recall = high-demand days detected; "
                    "F1 = balance between precision and recall."
                )

                left, right = st.columns(2)

                with left:
                    st.markdown("**Confusion matrix — validation evidence**")
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual: Normal", "Actual: High Demand"],
                        columns=["Predicted: Normal", "Predicted: High Demand"],
                    )
                    st.dataframe(cm_df, use_container_width=True)

                with right:
                    cm_long = pd.DataFrame([
                        {"Actual": "Normal", "Predicted": "Normal", "Count": int(tn)},
                        {"Actual": "Normal", "Predicted": "High Demand", "Count": int(fp)},
                        {"Actual": "High Demand", "Predicted": "Normal", "Count": int(fn)},
                        {"Actual": "High Demand", "Predicted": "High Demand", "Count": int(tp)},
                    ])
                    chart(
                        px.bar(
                            cm_long,
                            x="Actual",
                            y="Count",
                            color="Predicted",
                            barmode="group",
                            title="Confusion matrix",
                        ),
                        360,
                    )

                classification_table = pd.DataFrame([{
                    "Threshold": round(threshold, 2),
                    "Accuracy": round(accuracy, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1 Score": round(f1, 4),
                    "True Negatives": int(tn),
                    "False Positives": int(fp),
                    "False Negatives": int(fn),
                    "True Positives": int(tp),
                }])

                st.markdown("**Classification validation table**")
                st.dataframe(classification_table, use_container_width=True, hide_index=True)

                st.info(
                    "Note: MAE, RMSE and MAPE are the primary metrics because Demand is "
                    "a continuous forecasting target. Accuracy, Precision, Recall and "
                    "F1 are supplementary diagnostics for the business task of detecting "
                    "high-demand days."
                )

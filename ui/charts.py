# ui/charts.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# COMMON CHART SETTINGS
# ============================================================

CHART_HEIGHT = 360

BASE_LAYOUT = {
    "height": CHART_HEIGHT,
    "margin": dict(l=10, r=10, t=45, b=10),
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "font": dict(
        family="Inter, Arial, sans-serif",
        size=12,
    ),
    "hovermode": "x unified",
    "legend": dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
}


def apply_layout(fig, title=None, height=CHART_HEIGHT):
    """
    Apply the common enterprise dashboard layout.
    """

    layout = BASE_LAYOUT.copy()

    layout["height"] = height

    if title:
        layout["title"] = dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(
                size=16,
                color="#111827",
            ),
        )

    fig.update_layout(**layout)

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#e5e7eb",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
        linecolor="#e5e7eb",
    )

    return fig


# ============================================================
# 1. HISTORICAL VS FORECAST
# ============================================================

def historical_vs_forecast_chart(
    historical_df,
    forecast_df,
    title="Historical Demand vs Forecast",
):
    """
    Line chart comparing historical demand with future forecast.
    """

    fig = go.Figure()

    if historical_df is not None and not historical_df.empty:

        hist = historical_df.copy()

        hist["Date"] = pd.to_datetime(hist["Date"])

        if "Actual Demand" in hist.columns:

            daily_hist = (
                hist.groupby("Date", as_index=False)["Actual Demand"]
                .sum()
            )

            fig.add_trace(
                go.Scatter(
                    x=daily_hist["Date"],
                    y=daily_hist["Actual Demand"],
                    mode="lines",
                    name="Historical Demand",
                    line=dict(width=2),
                )
            )

    if forecast_df is not None and not forecast_df.empty:

        fcst = forecast_df.copy()

        fcst["Date"] = pd.to_datetime(fcst["Date"])

        if "Predicted Demand" in fcst.columns:

            daily_forecast = (
                fcst.groupby("Date", as_index=False)["Predicted Demand"]
                .sum()
            )

            fig.add_trace(
                go.Scatter(
                    x=daily_forecast["Date"],
                    y=daily_forecast["Predicted Demand"],
                    mode="lines",
                    name="Forecast Demand",
                    line=dict(width=2, dash="dash"),
                )
            )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 2. STORE DEMAND COMPARISON
# ============================================================

def store_demand_comparison(
    store_df,
    title="Demand Comparison by Store",
):
    """
    Compare historical and forecast demand across stores.
    """

    df = store_df.copy()

    required = {
        "Store ID",
        "Historical Demand",
        "Forecast Demand",
    }

    if not required.issubset(df.columns):

        # Support analytics output naming.
        rename_map = {}

        if "historical_demand" in df.columns:
            rename_map["historical_demand"] = "Historical Demand"

        if "forecast_7d" in df.columns:
            rename_map["forecast_7d"] = "Forecast Demand"

        df = df.rename(columns=rename_map)

    if not {
        "Store ID",
        "Historical Demand",
        "Forecast Demand",
    }.issubset(df.columns):

        return empty_chart("Store demand data unavailable")

    plot_df = df[
        [
            "Store ID",
            "Historical Demand",
            "Forecast Demand",
        ]
    ].copy()

    plot_df = plot_df.melt(
        id_vars="Store ID",
        var_name="Metric",
        value_name="Demand",
    )

    fig = px.bar(
        plot_df,
        x="Store ID",
        y="Demand",
        color="Metric",
        barmode="group",
        title=title,
    )

    return apply_layout(fig, title=title)


# ============================================================
# 3. INVENTORY VS FORECAST DEMAND
# ============================================================

def inventory_vs_forecast_chart(
    df,
    title="Inventory vs Expected 7-Day Demand",
):
    """
    Compare current inventory with forecast demand.
    """

    data = df.copy()

    rename_map = {}

    if "inventory" in data.columns:
        rename_map["inventory"] = "Inventory"

    if "forecast_7d" in data.columns:
        rename_map["forecast_7d"] = "Forecast Demand"

    if "forecast_demand" in data.columns:
        rename_map["forecast_demand"] = "Forecast Demand"

    data = data.rename(columns=rename_map)

    if "Inventory" not in data.columns:
        return empty_chart("Inventory data unavailable")

    if "Forecast Demand" not in data.columns:
        return empty_chart("Forecast demand unavailable")

    if "Store ID" in data.columns:

        grouped = (
            data.groupby("Store ID", as_index=False)
            .agg(
                Inventory=("Inventory", "sum"),
                **{
                    "Forecast Demand": (
                        "Forecast Demand",
                        "sum",
                    )
                },
            )
        )

        plot_df = grouped

        x_column = "Store ID"

    elif "Product ID" in data.columns:

        grouped = (
            data.groupby("Product ID", as_index=False)
            .agg(
                Inventory=("Inventory", "sum"),
                **{
                    "Forecast Demand": (
                        "Forecast Demand",
                        "sum",
                    )
                },
            )
        )

        plot_df = grouped

        x_column = "Product ID"

    else:

        plot_df = pd.DataFrame(
            {
                "Metric": [
                    "Inventory",
                    "Forecast Demand",
                ],
                "Value": [
                    data["Inventory"].sum(),
                    data["Forecast Demand"].sum(),
                ],
            }
        )

        fig = px.bar(
            plot_df,
            x="Metric",
            y="Value",
            title=title,
        )

        return apply_layout(fig, title=title)

    plot_df = plot_df.melt(
        id_vars=x_column,
        var_name="Metric",
        value_name="Units",
    )

    fig = px.bar(
        plot_df,
        x=x_column,
        y="Units",
        color="Metric",
        barmode="group",
        title=title,
    )

    return apply_layout(fig, title=title)


# ============================================================
# 4. RISK DISTRIBUTION
# ============================================================

def risk_distribution_chart(
    recommendations_df,
    title="Inventory Risk Distribution",
):
    """
    Show High / Medium / Low risk counts.
    """

    df = recommendations_df.copy()

    risk_column = None

    for column in [
        "Risk Level",
        "Risk",
        "risk_level",
    ]:

        if column in df.columns:
            risk_column = column
            break

    if risk_column is None:
        return empty_chart("Risk data unavailable")

    counts = (
        df[risk_column]
        .astype(str)
        .str.title()
        .value_counts()
        .reindex(
            ["High", "Medium", "Low"],
            fill_value=0,
        )
        .reset_index()
    )

    counts.columns = [
        "Risk Level",
        "Count",
    ]

    fig = px.bar(
        counts,
        x="Risk Level",
        y="Count",
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 5. TOP RISK PRODUCTS
# ============================================================

def top_risk_products_chart(
    recommendations_df,
    n=10,
    title="Top Inventory Risk Products",
):
    """
    Horizontal ranking of the highest-risk Store/Product pairs.
    """

    df = recommendations_df.copy()

    if df.empty:
        return empty_chart("No risk data available")

    if "Risk Score" in df.columns:

        df = df.sort_values(
            "Risk Score",
            ascending=False,
        )

        score_column = "Risk Score"

    elif "Projected Shortage" in df.columns:

        df = df.sort_values(
            "Projected Shortage",
            ascending=False,
        )

        score_column = "Projected Shortage"

    elif "Days Until Stockout" in df.columns:

        df = df.sort_values(
            "Days Until Stockout",
            ascending=True,
        )

        score_column = "Days Until Stockout"

    else:

        return empty_chart("Risk ranking unavailable")

    df = df.head(n).copy()

    if {
        "Store ID",
        "Product ID",
    }.issubset(df.columns):

        df["Item"] = (
            df["Store ID"].astype(str)
            + " / "
            + df["Product ID"].astype(str)
        )

    elif "Product ID" in df.columns:

        df["Item"] = df["Product ID"].astype(str)

    else:

        df["Item"] = df.index.astype(str)

    fig = px.bar(
        df.sort_values(score_column),
        x=score_column,
        y="Item",
        orientation="h",
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
        height=max(360, n * 38),
    )


# ============================================================
# 6. MONTHLY DEMAND TREND
# ============================================================

def monthly_demand_chart(
    monthly_df,
    title="Monthly Demand Trend",
):
    """
    Display monthly demand over time.
    """

    df = monthly_df.copy()

    if df.empty:
        return empty_chart("Monthly demand unavailable")

    if "Date" in df.columns:

        date_column = "Date"

    elif "Month" in df.columns:

        date_column = "Month"

    elif "month" in df.columns:

        date_column = "month"

    else:

        return empty_chart("Month column unavailable")

    if "Demand" in df.columns:

        demand_column = "Demand"

    elif "Historical Demand" in df.columns:

        demand_column = "Historical Demand"

    elif "demand" in df.columns:

        demand_column = "demand"

    else:

        return empty_chart("Demand column unavailable")

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[date_column]
    ).sort_values(date_column)

    fig = px.line(
        df,
        x=date_column,
        y=demand_column,
        markers=True,
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 7. PRODUCT RANKING
# ============================================================

def product_ranking_chart(
    product_df,
    n=10,
    title="Top Products by Historical Demand",
):
    """
    Rank products by historical demand.
    """

    df = product_df.copy()

    if df.empty:
        return empty_chart("Product data unavailable")

    if "Historical Demand" in df.columns:

        value_column = "Historical Demand"

    elif "historical_demand" in df.columns:

        value_column = "historical_demand"

    elif "Demand" in df.columns:

        value_column = "Demand"

    else:

        return empty_chart("Demand column unavailable")

    product_column = None

    for column in [
        "Product ID",
        "Product",
        "product_id",
    ]:

        if column in df.columns:
            product_column = column
            break

    if product_column is None:
        return empty_chart("Product identifier unavailable")

    df = (
        df.sort_values(
            value_column,
            ascending=False,
        )
        .head(n)
        .copy()
    )

    fig = px.bar(
        df.sort_values(value_column),
        x=value_column,
        y=product_column,
        orientation="h",
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
        height=max(360, n * 38),
    )


# ============================================================
# 8. REPLENISHMENT BY STORE
# ============================================================

def replenishment_by_store_chart(
    store_df,
    title="Recommended Replenishment by Store",
):
    """
    Compare recommended order quantities across stores.
    """

    df = store_df.copy()

    if df.empty:
        return empty_chart("Replenishment data unavailable")

    if "Recommended Order" in df.columns:

        order_column = "Recommended Order"

    elif "recommended_units" in df.columns:

        order_column = "recommended_units"

    elif "Recommended Units" in df.columns:

        order_column = "Recommended Units"

    else:

        return empty_chart("Recommended order data unavailable")

    if "Store ID" not in df.columns:
        return empty_chart("Store information unavailable")

    grouped = (
        df.groupby("Store ID", as_index=False)[order_column]
        .sum()
        .sort_values(
            order_column,
            ascending=False,
        )
    )

    fig = px.bar(
        grouped,
        x="Store ID",
        y=order_column,
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 9. INVENTORY COVERAGE
# ============================================================

def inventory_coverage_chart(
    df,
    title="Inventory Coverage by Store",
):
    """
    Show inventory coverage ratio by store.
    """

    data = df.copy()

    if "coverage_days" in data.columns:

        value_column = "coverage_days"

    elif "Coverage Days" in data.columns:

        value_column = "Coverage Days"

    elif "coverage" in data.columns:

        value_column = "coverage"

    else:

        return empty_chart(
            "Inventory coverage unavailable"
        )

    if "Store ID" not in data.columns:
        return empty_chart("Store information unavailable")

    plot_df = (
        data[
            [
                "Store ID",
                value_column,
            ]
        ]
        .drop_duplicates("Store ID")
        .sort_values(value_column)
    )

    fig = px.bar(
        plot_df,
        x=value_column,
        y="Store ID",
        orientation="h",
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 10. DEMAND VS INVENTORY SCATTER
# ============================================================

def demand_inventory_scatter(
    df,
    title="Demand vs Inventory",
):
    """
    Scatter plot showing inventory against expected demand.
    """

    data = df.copy()

    if "Inventory Level" in data.columns:
        inventory_column = "Inventory Level"

    elif "Inventory" in data.columns:
        inventory_column = "Inventory"

    elif "inventory" in data.columns:
        inventory_column = "inventory"

    else:
        return empty_chart("Inventory data unavailable")

    if "Forecast Demand" in data.columns:
        demand_column = "Forecast Demand"

    elif "forecast_7d" in data.columns:
        demand_column = "forecast_7d"

    elif "Forecast Demand 7D" in data.columns:
        demand_column = "Forecast Demand 7D"

    else:
        return empty_chart("Forecast demand unavailable")

    color_column = None

    for column in [
        "Risk Level",
        "Risk",
        "risk_level",
    ]:

        if column in data.columns:
            color_column = column
            break

    hover_columns = []

    for column in [
        "Store ID",
        "Product ID",
        "Risk Level",
        "Projected Shortage",
    ]:

        if column in data.columns:
            hover_columns.append(column)

    fig = px.scatter(
        data,
        x=demand_column,
        y=inventory_column,
        color=color_column,
        hover_data=hover_columns,
        title=title,
    )

    # Reference line: inventory = forecast demand.
    max_value = max(
        data[inventory_column].max(),
        data[demand_column].max(),
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            name="Balanced Inventory",
            line=dict(
                dash="dash",
                width=1,
            ),
        )
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 11. ACTUAL VS PREDICTED
# ============================================================

def actual_vs_predicted_chart(
    validation_df,
    title="Actual vs Predicted Demand",
):
    """
    Compare actual and predicted demand during validation.
    """

    df = validation_df.copy()

    if df.empty:
        return empty_chart("Validation data unavailable")

    if "Actual Demand" not in df.columns:
        return empty_chart("Actual demand unavailable")

    if "Predicted Demand" not in df.columns:
        return empty_chart("Predicted demand unavailable")

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    daily = (
        df.groupby("Date", as_index=False)
        .agg(
            Actual=("Actual Demand", "sum"),
            Predicted=("Predicted Demand", "sum"),
        )
        .sort_values("Date")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Actual"],
            mode="lines",
            name="Actual",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Predicted"],
            mode="lines",
            name="Predicted",
            line=dict(
                width=2,
                dash="dash",
            ),
        )
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 12. MODEL VS BASELINE
# ============================================================

def model_vs_baseline_chart(
    model_metrics,
    baseline_metrics,
    title="Model vs Baseline",
):
    """
    Compare forecasting errors.
    """

    metrics = [
        "MAE",
        "RMSE",
        "MAPE",
    ]

    rows = []

    for metric in metrics:

        if metric in model_metrics:

            rows.append(
                {
                    "Metric": metric,
                    "Model": "LightGBM",
                    "Value": model_metrics[metric],
                }
            )

        if metric in baseline_metrics:

            rows.append(
                {
                    "Metric": metric,
                    "Model": "Naive Baseline",
                    "Value": baseline_metrics[metric],
                }
            )

    if not rows:
        return empty_chart("Model metrics unavailable")

    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

def feature_importance_chart(
    importance_df,
    n=12,
    title="Top Forecasting Features",
):
    """
    Horizontal feature importance ranking.
    """

    df = importance_df.copy()

    if df.empty:
        return empty_chart("Feature importance unavailable")

    if "Feature" in df.columns:
        feature_column = "Feature"

    elif "feature" in df.columns:
        feature_column = "feature"

    else:
        return empty_chart("Feature names unavailable")

    if "Importance" in df.columns:
        importance_column = "Importance"

    elif "importance" in df.columns:
        importance_column = "importance"

    else:
        return empty_chart("Importance values unavailable")

    df = (
        df.sort_values(
            importance_column,
            ascending=False,
        )
        .head(n)
        .copy()
    )

    fig = px.bar(
        df.sort_values(importance_column),
        x=importance_column,
        y=feature_column,
        orientation="h",
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
        height=max(380, n * 35),
    )


# ============================================================
# 14. RISK BY STORE
# ============================================================

def risk_by_store_chart(
    recommendations_df,
    title="High-Risk Products by Store",
):
    """
    Count high-risk Store/Product combinations by store.
    """

    df = recommendations_df.copy()

    if df.empty or "Store ID" not in df.columns:
        return empty_chart("Store risk data unavailable")

    risk_column = None

    for column in [
        "Risk Level",
        "Risk",
        "risk_level",
    ]:

        if column in df.columns:
            risk_column = column
            break

    if risk_column is None:
        return empty_chart("Risk level unavailable")

    high_risk = df[
        df[risk_column]
        .astype(str)
        .str.lower()
        .eq("high")
    ]

    counts = (
        high_risk.groupby("Store ID")
        .size()
        .reset_index(name="High Risk Products")
        .sort_values(
            "High Risk Products",
            ascending=False,
        )
    )

    fig = px.bar(
        counts,
        x="Store ID",
        y="High Risk Products",
        title=title,
    )

    return apply_layout(
        fig,
        title=title,
    )


# ============================================================
# 15. EMPTY CHART
# ============================================================

def empty_chart(message="No data available"):
    """
    Return a clean empty Plotly chart instead of crashing
    the dashboard.
    """

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=14,
            color="#6b7280",
        ),
    )

    fig.update_layout(
        height=300,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            visible=False
        ),
        yaxis=dict(
            visible=False
        ),
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
    )

    return fig
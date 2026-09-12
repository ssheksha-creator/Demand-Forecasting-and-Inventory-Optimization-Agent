# ui/components.py

import pandas as pd
import streamlit as st

from ui.styles import render_risk_badge


# ============================================================
# PAGE HEADER
# ============================================================

def page_header(title, description=None, badge=None):
    """
    Render a consistent page header.
    """

    if badge:
        st.markdown(
            f'<div class="page-badge">{badge}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<div class="page-description">{description}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# KPI ROW
# ============================================================

def kpi_row(kpis):
    """
    Render a row of KPI cards.

    Expected format:

    [
        {
            "label": "Total Inventory",
            "value": "29,049",
            "caption": "Current stock",
            "variant": "blue"
        }
    ]
    """

    columns = st.columns(len(kpis))

    for column, item in zip(columns, kpis):

        with column:

            variant = item.get(
                "variant",
                "blue",
            )

            st.markdown(
                f"""
                <div class="kpi-card kpi-{variant}">

                    <div class="kpi-label">
                        {item.get("label", "")}
                    </div>

                    <div class="kpi-value">
                        {item.get("value", "")}
                    </div>

                    <div class="kpi-caption">
                        {item.get("caption", "")}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# SECTION CARD
# ============================================================

def section_start(
    title,
    subtitle=None,
):
    """
    Start a visual section.
    """

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:

        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def section_end():
    """
    Close a visual section.
    """

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# RISK BADGE
# ============================================================

def risk_badge(risk):
    """
    Return HTML risk badge.
    """

    return render_risk_badge(risk)


# ============================================================
# RISK SUMMARY
# ============================================================

def risk_summary(
    high,
    medium,
    low,
):
    """
    Render High / Medium / Low risk summary cards.
    """

    columns = st.columns(3)

    risk_data = [
        ("High Risk", high, "red"),
        ("Medium Risk", medium, "amber"),
        ("Low Risk", low, "green"),
    ]

    for column, (label, value, variant) in zip(
        columns,
        risk_data,
    ):

        with column:

            st.markdown(
                f"""
                <div class="kpi-card kpi-{variant}">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value:,}
                    </div>

                    <div class="kpi-caption">
                        Store-product combinations
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ATTENTION REQUIRED TABLE
# ============================================================

def attention_required_table(
    recommendations,
    n=10,
):
    """
    Render the highest-priority inventory actions.
    """

    if recommendations is None or recommendations.empty:

        st.info(
            "No inventory risks require attention."
        )

        return

    df = recommendations.copy()

    # Sort by stockout timing first.
    if "Days Until Stockout" in df.columns:

        df = df.sort_values(
            "Days Until Stockout",
            ascending=True,
        )

    elif "Projected Shortage" in df.columns:

        df = df.sort_values(
            "Projected Shortage",
            ascending=False,
        )

    df = df.head(n).copy()

    # Create display-friendly item name.
    if {
        "Store ID",
        "Product ID",
    }.issubset(df.columns):

        df["Item"] = (
            df["Store ID"].astype(str)
            + " / "
            + df["Product ID"].astype(str)
        )

    display_columns = []

    if "Item" in df.columns:
        display_columns.append("Item")

    risk_column = None

    for column in [
        "Risk Level",
        "Risk",
        "risk_level",
    ]:

        if column in df.columns:
            risk_column = column
            break

    if risk_column:
        display_columns.append(risk_column)

    for column in [
        "Current Inventory",
        "Inventory Level",
        "Recommended Order",
        "Days Until Stockout",
        "Projected Shortage",
    ]:

        if column in df.columns:
            display_columns.append(column)

    if not display_columns:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        return

    display_df = df[display_columns].copy()

    # Rename columns for presentation.
    rename_map = {
        "Current Inventory": "Current Stock",
        "Inventory Level": "Current Stock",
        "Recommended Order": "Recommended Order",
        "Days Until Stockout": "Days to Stockout",
        "Projected Shortage": "Projected Shortage",
        risk_column: "Risk",
    }

    display_df = display_df.rename(
        columns=rename_map
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ACTION TABLE
# ============================================================

def action_table(
    dataframe,
    columns=None,
    height=420,
):
    """
    Generic polished dataframe renderer.
    """

    if dataframe is None or dataframe.empty:

        st.info("No data available.")

        return

    df = dataframe.copy()

    if columns:

        available = [
            column
            for column in columns
            if column in df.columns
        ]

        if available:
            df = df[available]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


# ============================================================
# SELECTOR ROW
# ============================================================

def product_store_selector(
    data,
    key_prefix="selector",
):
    """
    Create Store and Product selectors.

    Returns:
        store_id, product_id
    """

    stores = sorted(
        data["Store ID"]
        .dropna()
        .astype(str)
        .unique()
    )

    products = sorted(
        data["Product ID"]
        .dropna()
        .astype(str)
        .unique()
    )

    col1, col2 = st.columns(2)

    with col1:

        store_id = st.selectbox(
            "Store",
            stores,
            key=f"{key_prefix}_store",
        )

    with col2:

        product_id = st.selectbox(
            "Product",
            products,
            key=f"{key_prefix}_product",
        )

    return store_id, product_id


# ============================================================
# FILTER ROW
# ============================================================

def filter_row(
    data,
    key_prefix="filters",
):
    """
    Dashboard-wide Store / Product / Category filters.
    """

    col1, col2, col3 = st.columns(3)

    stores = sorted(
        data["Store ID"]
        .dropna()
        .astype(str)
        .unique()
    )

    products = sorted(
        data["Product ID"]
        .dropna()
        .astype(str)
        .unique()
    )

    categories = sorted(
        data["Category"]
        .dropna()
        .astype(str)
        .unique()
    )

    with col1:

        selected_store = st.selectbox(
            "Store",
            ["All"] + stores,
            key=f"{key_prefix}_store",
        )

    with col2:

        selected_product = st.selectbox(
            "Product",
            ["All"] + products,
            key=f"{key_prefix}_product",
        )

    with col3:

        selected_category = st.selectbox(
            "Category",
            ["All"] + categories,
            key=f"{key_prefix}_category",
        )

    filtered = data.copy()

    if selected_store != "All":

        filtered = filtered[
            filtered["Store ID"].astype(str)
            == selected_store
        ]

    if selected_product != "All":

        filtered = filtered[
            filtered["Product ID"].astype(str)
            == selected_product
        ]

    if selected_category != "All":

        filtered = filtered[
            filtered["Category"].astype(str)
            == selected_category
        ]

    return filtered


# ============================================================
# INSIGHT CARD
# ============================================================

def insight_card(
    title,
    message,
    icon="💡",
):
    """
    Render an insight card.
    """

    st.markdown(
        f"""
        <div class="action-card">

            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                margin-bottom:8px;
            ">

                <span style="font-size:18px;">
                    {icon}
                </span>

                <div class="action-title">
                    {title}
                </div>

            </div>

            <div class="action-text">
                {message}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RECOMMENDATION CARD
# ============================================================

def recommendation_card(
    product,
    store,
    risk,
    current_inventory,
    forecast_demand,
    recommended_order,
    days_until_stockout=None,
):
    """
    Display a single actionable inventory recommendation.
    """

    badge = risk_badge(risk)

    stockout_text = ""

    if days_until_stockout is not None:

        stockout_text = (
            f"""
            <div class="metric-highlight">

                <div class="metric-highlight-label">
                    Days to Stockout
                </div>

                <div class="metric-highlight-value">
                    {days_until_stockout:.1f}
                </div>

            </div>
            """
        )

    st.markdown(
        f"""
        <div class="action-card">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:12px;
            ">

                <div>

                    <div class="action-title">
                        {product}
                    </div>

                    <div class="action-text">
                        Store {store}
                    </div>

                </div>

                <div>
                    {badge}
                </div>

            </div>

            <div style="
                display:grid;
                grid-template-columns:
                    repeat(3, 1fr);
                gap:10px;
            ">

                <div class="metric-highlight">

                    <div class="metric-highlight-label">
                        Current Stock
                    </div>

                    <div class="metric-highlight-value">
                        {current_inventory:,.0f}
                    </div>

                </div>

                <div class="metric-highlight">

                    <div class="metric-highlight-label">
                        7-Day Demand
                    </div>

                    <div class="metric-highlight-value">
                        {forecast_demand:,.0f}
                    </div>

                </div>

                <div class="metric-highlight">

                    <div class="metric-highlight-label">
                        Recommended Order
                    </div>

                    <div class="metric-highlight-value">
                        {recommended_order:,.0f}
                    </div>

                </div>

            </div>

            {stockout_text}

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AI EXPLANATION
# ============================================================

def ai_explanation(
    explanation,
    title="AI Recommendation",
):
    """
    Render an AI-generated business explanation.
    """

    st.markdown(
        f"""
        <div class="ai-card">

            <div class="ai-header">

                <div class="ai-icon">
                    🤖
                </div>

                <div>

                    <div class="ai-title">
                        {title}
                    </div>

                    <div class="ai-subtitle">
                        Grounded in forecast and inventory calculations
                    </div>

                </div>

            </div>

            <div class="ai-message">
                {explanation}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EMPTY STATE
# ============================================================

def empty_state(
    title="No data available",
    message="There is currently nothing to display.",
    icon="📊",
):
    """
    Render a clean empty-state component.
    """

    st.markdown(
        f"""
        <div class="section-card"
             style="text-align:center;padding:45px;">

            <div style="font-size:36px;">
                {icon}
            </div>

            <div class="section-title"
                 style="margin-top:12px;">
                {title}
            </div>

            <div class="section-subtitle">
                {message}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST SUMMARY
# ============================================================

def forecast_summary(
    current_inventory,
    forecast_demand,
    safety_stock,
    reorder_point,
    recommended_order,
):
    """
    Compact forecast/inventory summary.
    """

    columns = st.columns(5)

    values = [
        (
            "Current Inventory",
            current_inventory,
        ),
        (
            "7-Day Forecast",
            forecast_demand,
        ),
        (
            "Safety Stock",
            safety_stock,
        ),
        (
            "Reorder Point",
            reorder_point,
        ),
        (
            "Recommended Order",
            recommended_order,
        ),
    ]

    for column, (label, value) in zip(
        columns,
        values,
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-highlight">

                    <div class="metric-highlight-label">
                        {label}
                    </div>

                    <div class="metric-highlight-value">
                        {value:,.0f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# CHAT MESSAGE
# ============================================================

def chat_message(
    message,
    role="assistant",
):
    """
    Render a lightweight AI chat message.
    """

    if role == "user":

        background = "#eff6ff"
        label = "You"

    else:

        background = "#f8fafc"
        label = "AI Copilot"

    st.markdown(
        f"""
        <div style="
            background:{background};
            border-radius:12px;
            padding:12px 15px;
            margin:8px 0;
            border:1px solid #e5e7eb;
        ">

            <div style="
                font-size:10px;
                font-weight:700;
                color:#6b7280;
                margin-bottom:4px;
                text-transform:uppercase;
            ">
                {label}
            </div>

            <div style="
                font-size:13px;
                color:#374151;
                line-height:1.55;
            ">
                {message}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NUMBER FORMATTERS
# ============================================================

def format_number(value):
    """
    Format a number with commas.
    """

    if value is None:
        return "—"

    try:
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return str(value)


def format_decimal(value, decimals=1):
    """
    Format a decimal number.
    """

    if value is None:
        return "—"

    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_percent(value, decimals=1):
    """
    Format a percentage.
    """

    if value is None:
        return "—"

    try:
        return f"{float(value):.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)
# ui/styles.py

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

def configure_page():
    """
    Configure the Streamlit page.
    """

    st.set_page_config(
        page_title="AI Inventory Copilot",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ============================================================
# GLOBAL CSS
# ============================================================

def load_css():
    """
    Load the premium enterprise dashboard styling.
    """

    st.markdown(
        """
        <style>

        /* ====================================================
           GLOBAL
        ==================================================== */

        .stApp {
            background-color: #f6f8fb;
        }

        .main {
            padding-top: 1rem;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Remove Streamlit default decoration */

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }


        /* ====================================================
           SIDEBAR
        ==================================================== */

        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        section[data-testid="stSidebar"] > div {
            background-color: #111827;
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        .sidebar-brand {
            padding: 8px 8px 22px 8px;
        }

        .sidebar-logo {
            width: 46px;
            height: 46px;
            border-radius: 13px;
            background: #2563eb;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
            margin-bottom: 12px;
        }

        .sidebar-title {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 3px;
        }

        .sidebar-subtitle {
            font-size: 12px;
            color: #9ca3af;
        }

        .sidebar-section {
            font-size: 11px;
            font-weight: 700;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 18px 8px 8px 8px;
        }


        /* ====================================================
           PAGE HEADER
        ==================================================== */

        .page-header {
            margin-bottom: 24px;
        }

        .page-title {
            font-size: 31px;
            font-weight: 750;
            color: #111827;
            line-height: 1.15;
            margin-bottom: 6px;
        }

        .page-description {
            color: #6b7280;
            font-size: 14px;
            line-height: 1.5;
        }

        .page-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            background: #eff6ff;
            color: #2563eb;
            font-size: 11px;
            font-weight: 700;
            margin-bottom: 10px;
        }


        /* ====================================================
           KPI CARDS
        ==================================================== */

        .kpi-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 15px;
            padding: 20px;
            min-height: 132px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
        }

        .kpi-label {
            font-size: 12px;
            color: #6b7280;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .kpi-value {
            font-size: 29px;
            font-weight: 750;
            color: #111827;
            line-height: 1.1;
            margin-bottom: 8px;
        }

        .kpi-caption {
            font-size: 11px;
            color: #9ca3af;
        }

        .kpi-blue {
            border-top: 3px solid #2563eb;
        }

        .kpi-red {
            border-top: 3px solid #dc2626;
        }

        .kpi-green {
            border-top: 3px solid #16a34a;
        }

        .kpi-amber {
            border-top: 3px solid #d97706;
        }


        /* ====================================================
           SECTION CARDS
        ==================================================== */

        .section-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
        }

        .section-title {
            color: #111827;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .section-subtitle {
            color: #6b7280;
            font-size: 12px;
            margin-bottom: 16px;
        }


        /* ====================================================
           RISK BADGES
        ==================================================== */

        .risk-high {
            display: inline-block;
            background: #fef2f2;
            color: #dc2626;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
        }

        .risk-medium {
            display: inline-block;
            background: #fffbeb;
            color: #d97706;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
        }

        .risk-low {
            display: inline-block;
            background: #f0fdf4;
            color: #16a34a;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
        }


        /* ====================================================
           STATUS / INFO BOXES
        ==================================================== */

        .info-box {
            background: #eff6ff;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 14px 16px;
            color: #1e40af;
            font-size: 13px;
            line-height: 1.5;
        }

        .warning-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 12px;
            padding: 14px 16px;
            color: #92400e;
            font-size: 13px;
            line-height: 1.5;
        }

        .danger-box {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 12px;
            padding: 14px 16px;
            color: #991b1b;
            font-size: 13px;
            line-height: 1.5;
        }

        .success-box {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 14px 16px;
            color: #166534;
            font-size: 13px;
            line-height: 1.5;
        }


        /* ====================================================
           AI COPILOT
        ==================================================== */

        .ai-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 3px 12px rgba(37, 99, 235, 0.06);
        }

        .ai-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }

        .ai-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: #eff6ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .ai-title {
            color: #111827;
            font-size: 16px;
            font-weight: 700;
        }

        .ai-subtitle {
            color: #6b7280;
            font-size: 11px;
        }

        .ai-message {
            background: #f8fafc;
            border-radius: 12px;
            padding: 13px;
            margin-top: 10px;
            color: #374151;
            font-size: 13px;
            line-height: 1.55;
        }


        /* ====================================================
           ACTION CARDS
        ==================================================== */

        .action-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 13px;
            padding: 16px;
            transition: all 0.15s ease;
        }

        .action-card:hover {
            border-color: #bfdbfe;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
        }

        .action-title {
            font-size: 14px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 5px;
        }

        .action-text {
            font-size: 12px;
            color: #6b7280;
            line-height: 1.5;
        }


        /* ====================================================
           METRIC HIGHLIGHT
        ==================================================== */

        .metric-highlight {
            background: #f8fafc;
            border-radius: 12px;
            padding: 14px;
            text-align: center;
        }

        .metric-highlight-label {
            font-size: 10px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-highlight-value {
            font-size: 22px;
            font-weight: 750;
            color: #111827;
            margin-top: 4px;
        }


        /* ====================================================
           STREAMLIT INPUTS
        ==================================================== */

        div[data-baseweb="select"] > div {
            border-radius: 9px;
            border-color: #d1d5db;
        }

        div[data-baseweb="input"] > div {
            border-radius: 9px;
            border-color: #d1d5db;
        }

        .stTextInput input,
        .stNumberInput input {
            border-radius: 9px;
        }

        .stSelectbox label,
        .stMultiSelect label,
        .stNumberInput label,
        .stTextInput label,
        .stSlider label {
            color: #374151 !important;
            font-size: 12px !important;
            font-weight: 600 !important;
        }


        /* ====================================================
           BUTTONS
        ==================================================== */

        .stButton > button {
            border-radius: 9px;
            border: 1px solid #d1d5db;
            font-weight: 600;
            font-size: 13px;
            min-height: 40px;
            transition: all 0.15s ease;
        }

        .stButton > button:hover {
            border-color: #2563eb;
            color: #2563eb;
        }


        /* ====================================================
           DATAFRAME
        ==================================================== */

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] th {
            font-size: 12px;
        }


        /* ====================================================
           TABS
        ==================================================== */

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #e5e7eb;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 13px;
            font-weight: 600;
        }


        /* ====================================================
           DIVIDER
        ==================================================== */

        hr {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 22px 0;
        }


        /* ====================================================
           FOOTER
        ==================================================== */

        .app-footer {
            text-align: center;
            padding: 30px 0 10px 0;
            color: #9ca3af;
            font-size: 11px;
        }


        /* ====================================================
           RESPONSIVE
        ==================================================== */

        @media (max-width: 900px) {

            .page-title {
                font-size: 25px;
            }

            .kpi-value {
                font-size: 24px;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR BRAND
# ============================================================

def render_sidebar_brand():

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                📦
            </div>

            <div class="sidebar-title">
                AI Inventory Copilot
            </div>

            <div class="sidebar-subtitle">
                Demand Forecasting & Inventory Optimization
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE HEADER
# ============================================================

def render_page_header(
    title,
    description=None,
    badge=None
):

    html = ""

    if badge:
        html += f"""
        <div class="page-badge">
            {badge}
        </div>
        """

    html += f"""
        <div class="page-header">

            <div class="page-title">
                {title}
            </div>
    """

    if description:
        html += f"""
            <div class="page-description">
                {description}
            </div>
        """

    html += """
        </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# SECTION HEADER
# ============================================================

def render_section_header(
    title,
    subtitle=None
):

    html = f"""
    <div class="section-title">
        {title}
    </div>
    """

    if subtitle:
        html += f"""
        <div class="section-subtitle">
            {subtitle}
        </div>
        """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# KPI CARD
# ============================================================

def render_kpi(
    label,
    value,
    caption="",
    variant="blue"
):

    variant_class = f"kpi-{variant}"

    st.markdown(
        f"""
        <div class="kpi-card {variant_class}">

            <div class="kpi-label">
                {label}
            </div>

            <div class="kpi-value">
                {value}
            </div>

            <div class="kpi-caption">
                {caption}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RISK BADGE
# ============================================================

def render_risk_badge(risk):

    risk = str(risk).strip()

    if risk.lower() == "high":

        return """
        <span class="risk-high">
            HIGH
        </span>
        """

    if risk.lower() == "medium":

        return """
        <span class="risk-medium">
            MEDIUM
        </span>
        """

    return """
    <span class="risk-low">
        LOW
    </span>
    """


# ============================================================
# INFO BOX
# ============================================================

def render_info_box(
    message,
    box_type="info"
):

    classes = {
        "info": "info-box",
        "warning": "warning-box",
        "danger": "danger-box",
        "success": "success-box",
    }

    css_class = classes.get(
        box_type,
        "info-box"
    )

    st.markdown(
        f"""
        <div class="{css_class}">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AI CARD
# ============================================================

def render_ai_card(
    title="AI Inventory Copilot",
    subtitle="Decision support powered by your forecasting and inventory data."
):

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
                        {subtitle}
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ACTION CARD
# ============================================================

def render_action_card(
    title,
    description
):

    st.markdown(
        f"""
        <div class="action-card">

            <div class="action-title">
                {title}
            </div>

            <div class="action-text">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# METRIC HIGHLIGHT
# ============================================================

def render_metric_highlight(
    label,
    value
):

    st.markdown(
        f"""
        <div class="metric-highlight">

            <div class="metric-highlight-label">
                {label}
            </div>

            <div class="metric-highlight-value">
                {value}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.markdown(
        """
        <div class="app-footer">
            AI Inventory Copilot · Demand Forecasting &
            Inventory Optimization · Built for Enterprise
            Decision Support
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# COMPLETE UI INITIALIZATION
# ============================================================

def initialize_ui():

    configure_page()
    load_css()
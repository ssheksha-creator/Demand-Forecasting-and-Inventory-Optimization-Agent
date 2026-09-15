# Demand Forecasting & Inventory Optimization Agent

**Cognizant Hackathon 2026 – Use Case 4**  
**Team 38 | GITAM University**

An end-to-end AI-powered demand forecasting and inventory optimization solution that transforms historical retail data into demand forecasts, inventory-risk insights, replenishment recommendations, What-If scenarios, and AI-assisted business decisions.

---

## 📌 Use Case

### Use Case Description / Problem Statement

Retailers face stockouts and excess inventory due to inaccurate demand planning. Demand varies by store, product, region, season, promotion and price.

### Objective

Build a forecasting and inventory optimization solution that predicts demand, identifies stockout risk and recommends replenishment actions for human approval.

**Type:** Time Series Forecasting / Optimization

---

## 🎯 Solution Overview

**Historical Sales Data → Data Validation → Feature Engineering → Demand Forecasting → Forecast Uncertainty → Inventory Risk → Replenishment Optimization → AI Decision Support → Interactive Dashboard → Business Action**

The solution answers four key business questions:

1. What will demand look like?
2. Where is inventory at risk?
3. How much should be replenished?
4. Why is the recommendation being made?

---

## 📊 Dataset & Data Understanding

### Dataset Overview

- **Dataset:** Retail Store Inventory and Demand Forecasting
- **Records:** 76,000
- **Features:** 16
- **Time Period:** 01-Jan-2022 to 30-Jan-2024
- **Stores:** 5
- **Products:** 20
- **Store–Product Combinations:** 100
- **Daily Observations:** 760 per Store–Product series
- **Missing Values:** None
- **Primary Target:** `Demand`

### Data Parameters

| Category | Parameters |
|---|---|
| Time | Date |
| Store & Product | Store ID, Product ID, Category, Region |
| Inventory & Sales | Inventory Level, Units Ordered, Units Sold, Demand |
| Pricing | Price, Discount, Competitor Pricing |
| Business Factors | Promotion, Seasonality |
| External Factors | Weather Condition, Epidemic |

### Key Data Insights

- Demand varies across stores and products.
- Promotion and discount levels influence demand.
- Demand shows time-based and seasonal patterns.
- Inventory availability can affect the relationship between demand and units sold.
- Historical demand patterns provide the foundation for forecasting and inventory optimization.

---

## 🏗️ System Architecture

The solution is organized into five layers.

### Layer 1 – Data & Processing

- Load historical retail data from `sales_data.csv`.
- Validate data quality and structure.
- Generate time-series and business features.
- Prepare time-safe model inputs.
- Prevent future information from leaking into model features.

### Layer 2 – Demand Forecasting

- Global LightGBM forecasting model.
- Store and product identifiers are incorporated into the forecasting process.
- Uses historical demand lags and rolling statistics.
- Produces uncertainty-aware **P10, P50 and P90** demand forecasts.
- Uses chronological time-series validation.

### Layer 3 – Inventory Intelligence

The forecast is converted into inventory decisions using:

- Current inventory
- Lead time
- Safety stock
- Reorder point
- Target inventory
- Forecast uncertainty
- Days until stockout
- Projected shortage
- Stockout-risk classification

### Layer 4 – AI Decision Support

The AI Agent accepts natural-language business queries and generates relevant analysis.

Examples:

- Which products are at highest stockout risk?
- Compare P0018 and P0016 in Store S002.
- Which store needs the most replenishment?
- What happens if demand increases by 20%?
- How does promotion affect demand?

The Agent can independently generate relevant charts, tables, evidence and explanations.

### Layer 5 – Visualization & Business Action

The Streamlit application presents:

- Forecasts
- Inventory risk
- Replenishment recommendations
- What-If scenarios
- Store analytics
- Product analytics
- Model performance
- AI-generated analysis

---

## 🤖 Demand Forecasting Model

### Model

**LightGBM**

A global forecasting approach is used across Store × Product series rather than training an independent model for every series.

### Feature Engineering

The forecasting pipeline uses advanced time-series and business features, including:

- Demand lags: 1, 2, 3, 7, 14, 21, 28, 35, 56 and 90 days
- Rolling statistics over 7, 14, 28, 56 and 90 days
- Rolling mean, standard deviation, minimum and maximum
- Demand trend and momentum features
- Volatility ratios
- Recent-vs-historical demand ratios
- Historical expanding statistics
- Calendar features
- Cyclical time features
- Store and product information
- Business-related features available for forecasting

### Future-Safe Forecasting

Future-unknown operational variables are not directly used as future forecasting inputs:

- Inventory Level
- Units Ordered
- Units Sold
- Weather Condition
- Competitor Pricing
- Epidemic

This helps maintain a time-safe forecasting setup.

### Model Configuration

```text
Model: LightGBM
Point Forecast Estimators: 1800
Learning Rate: 0.025
Num Leaves: 63
Min Child Samples: 25
Subsample: 0.85
Column Sample by Tree: 0.85
L1 Regularization: 0.15
L2 Regularization: 0.25
```

### Forecast Output

- **P10:** Lower demand estimate
- **P50:** Central demand estimate
- **P90:** Higher demand estimate

---

## 📈 Model Validation

Validation uses **chronological time-series splits** rather than random train/test splitting.

### Rolling Validation

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **LightGBM V2** | **10.30** | **13.40** | **18.02%** |
| Moving Average 7D | 31.64 | 40.66 | 47.16% |
| Naive Yesterday | 39.69 | 51.39 | 53.80% |
| Seasonal Naive | 42.98 | 54.78 | 61.84% |

### Final 30-Day Holdout

- **MAE:** 9.48
- **RMSE:** 12.42
- **MAPE:** 15.00%

The model is evaluated against simple forecasting baselines to demonstrate predictive value.

---

## 📦 Inventory Optimization

The inventory layer converts demand forecasts into actionable replenishment decisions.

### Configuration

```text
Lead Time: 3 days
Service Level: 95%
Optimization Horizon: 7 days
```

### Safety Stock

```text
Safety Stock =
max(
    Statistical Safety Stock,
    Lead-Time P90 − Lead-Time P50
)
```

### Reorder Point

```text
Reorder Point =
Lead-Time P50 + Safety Stock
```

### Target Inventory

```text
Target Inventory =
max(
    7-Day P50 + Safety Stock,
    7-Day P90
)
```

### Recommended Order

```text
Recommended Order =
ceil(max(Target Inventory − Current Inventory, 0))
```

### Projected Shortage

```text
Projected Shortage =
max(P90 Demand − Current Inventory, 0)
```

### Stockout Risk

Risk is calculated using inventory position relative to forecast demand and is increased when estimated stockout occurs within the lead-time window.

```text
High Risk    ≥ 70
Medium Risk  35–69.99
Low Risk     < 35
```

---

## 🔮 What-If Simulator

The What-If Simulator evaluates demand scenarios before inventory decisions are taken.

```text
Baseline Forecast
      ↓
Apply Demand Change
      ↓
Recalculate P50 / P90
      ↓
Recalculate Shortage
      ↓
Recalculate Recommended Order
      ↓
Compare Scenario vs Baseline
```

Example:

> What happens if demand increases by 20%?

The simulator provides scenario-level demand, shortage and replenishment impacts.

---

## 🤖 AI Agent

The AI Agent provides natural-language access to analytical capabilities.

### Agent Workflow

**User Query → Intent Detection → Data Analysis → Relevant Charts → Evidence Tables → Explanation → Recommendation**

The Agent can independently generate:

- Charts
- Comparison tables
- Risk rankings
- Historical evidence
- Recent-demand evidence
- Explanations
- Business recommendations

### Supported Analysis

- Product comparison
- Store analytics
- Product analytics
- Stockout-risk analysis
- Promotion analysis
- Discount analysis
- Monthly/seasonal demand analysis
- What-If scenarios
- General dataset analysis

Numerical forecasts and inventory recommendations come from the underlying data, forecasting and optimization components.

---

## 🖥️ Application Pages

| Page | Purpose |
|---|---|
| **Dashboard** | Executive overview of demand, inventory and recommendations |
| **Demand Forecast** | View future demand forecasts and uncertainty |
| **Inventory Risk** | Identify stockout and inventory risks |
| **Replenishment** | View recommended order quantities and inventory actions |
| **What-If Simulator** | Evaluate demand-change scenarios |
| **AI Agent** | Ask natural-language business questions |
| **Store Analytics** | Analyze demand and inventory by store |
| **Product Analytics** | Analyze product-level demand and inventory |
| **Model Performance** | Evaluate forecasting performance and baselines |

---

## 🛠️ Technology Stack

### Programming & Data
- Python
- pandas
- NumPy

### Machine Learning
- LightGBM
- scikit-learn

### Visualization
- Plotly
- Matplotlib

### Application
- Streamlit

### AI
- OpenAI / LLM-based AI Agent integration

### Development
- PyCharm
- Git / GitHub

### Optional Future Deployment
- Docker
- Cloud infrastructure
- Enterprise ERP / inventory-system integration

---

## 📁 Project Structure

```text
AI_Demand_Inventory_Optimizer/
│
├── sales_data.csv
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── models/
│   ├── demand_model.pkl
│   └── model_metadata.json
│
├── outputs/
│   ├── forecasts.csv
│   └── recommendations.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── forecasting.py
│   ├── inventory.py
│   ├── analytics.py
│   └── agent.py
│
└── ui/
    ├── __init__.py
    ├── styles.py
    ├── components.py
    └── charts.py
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd AI_Demand_Inventory_Optimizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` with the required AI API configuration used by the project.

**Do not commit API keys or other secrets to GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

---

## 🔐 Data & Security Notes

- The project uses the provided `sales_data.csv` as its primary dataset.
- API credentials should be stored in environment variables.
- `.env` should not be committed to source control.
- Forecasting features are designed to avoid future operational information leakage.
- AI-generated explanations do not replace the underlying numerical forecasting and optimization calculations.

---

## 💡 Business Value

The solution provides:

- Proactive stockout identification
- Data-driven replenishment
- Safety-stock and reorder-point optimization
- High-risk product prioritization
- Demand scenario analysis
- AI-powered decision support
- Interactive forecast-to-action visibility

### Core Business Flow

**Forecast → Detect Risk → Optimize Inventory → Recommend Action → Explain Decision**

---

## 🚀 Roadmap

### Current Prototype

```text
Historical Data
      ↓
Forecasting
      ↓
Risk Detection
      ↓
Inventory Optimization
      ↓
AI Agent
      ↓
Interactive Dashboard
```

### Next Phase

- Integrate live sales and inventory feeds.
- Add automated forecast/replenishment alerts.
- Improve forecast accuracy with model monitoring and retraining.
- Integrate with ERP / inventory management systems.
- Add automated approval workflows for replenishment.
- Deploy on cloud infrastructure for enterprise-scale usage.

---

## 👥 Team 38 – GITAM University

| Team Member | Contribution |
|---|---|
| **Shaik Shekshavali – Team Leader** | **Developed the complete source code and integrated the entire solution end-to-end**, including data preprocessing, feature engineering, demand forecasting, model validation, inventory optimization, stockout-risk analysis, replenishment recommendations, What-If simulation, AI Agent, analytics modules, Streamlit dashboard, visualizations, model integration, testing, debugging and overall system integration. Also coordinated the team and contributed to the final presentation. |
| **Kesanapalli Lakshmi Sree** | Supported **inventory intelligence and optimization**, including inventory-risk analysis, safety-stock concepts, replenishment logic and interpretation of inventory optimization results. |
| **Yalama Reddy Mahesh Kumar Reddy** | Supported **What-If scenario analysis and business decision interpretation**, including analysis of demand changes and their impact on inventory and replenishment decisions. |
| **Madineni Hima Priya** | Supported **dashboard visualization and UI/UX review**, including feedback on charts, tables, dashboard layout and presentation of business insights. |

### Core Technical Contribution

> The complete application source code was developed and integrated by **Shaik Shekshavali**, covering the full pipeline from raw sales data to forecasting, inventory optimization, AI-assisted analysis and interactive visualization. Team members contributed through analysis, validation, domain interpretation, UI feedback, documentation and presentation activities.

---

## 🏆 Project Highlights

- End-to-end demand forecasting and inventory optimization workflow
- 76,000-record retail dataset
- 100 Store × Product forecasting series
- Advanced time-series feature engineering
- LightGBM global forecasting model
- P10/P50/P90 uncertainty-aware forecasting
- Chronological model validation
- Inventory risk classification
- Safety-stock and reorder-point optimization
- Replenishment recommendations
- What-If demand simulation
- Natural-language AI Agent
- Automatically generated analytical charts and evidence tables
- Interactive Streamlit application

---

## 📌 Key Takeaway

> **AI Demand Forecasting & Inventory Optimization Copilot converts historical retail data into actionable forecasts, identifies inventory risks, recommends replenishment actions and provides AI-assisted explanations to support faster and more informed business decisions.**

---

## 📄 Hackathon

**Cognizant Hackathon 2026**  
**Use Case 4 – Demand Forecasting & Inventory Optimization Agent**  
**Team 38 – GITAM University**

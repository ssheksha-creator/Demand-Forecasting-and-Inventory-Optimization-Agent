from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'sales_data.csv'
TARGET_COLUMN = 'Demand'
REQUIRED_COLUMNS = [
    'Date','Store ID','Product ID','Category','Region','Inventory Level',
    'Units Ordered','Units Sold','Demand','Price','Discount',
    'Weather Condition','Promotion','Seasonality','Epidemic','Competitor Pricing'
]
GROUP_KEYS = ['Store ID','Product ID']
DEMAND_LAGS = [1,2,3,7,14,21,28,35,56,90]
ROLLING_WINDOWS = [7,14,28,56,90]


def load_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isna().any():
        raise ValueError('Invalid Date values found.')
    return df.sort_values(GROUP_KEYS + ['Date']).reset_index(drop=True)


def _shift(df, col, periods=1):
    return df.groupby(GROUP_KEYS, sort=False)[col].shift(periods)


def _rolling(df, col, window, stat):
    # Shift BEFORE rolling: current target is never included.
    s = _shift(df, col, 1)
    g = s.groupby([df['Store ID'], df['Product ID']], sort=False)
    if stat == 'mean':
        out = g.transform(lambda x: x.rolling(window, min_periods=1).mean())
    elif stat == 'std':
        out = g.transform(lambda x: x.rolling(window, min_periods=2).std())
    elif stat == 'min':
        out = g.transform(lambda x: x.rolling(window, min_periods=1).min())
    elif stat == 'max':
        out = g.transform(lambda x: x.rolling(window, min_periods=1).max())
    else:
        raise ValueError(stat)
    return out


def _safe_div(a, b, fill=0.0):
    out = a.astype(float) / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def add_calendar_features(df):
    df = df.copy()
    d = df['Date']
    df['year'] = d.dt.year
    df['month'] = d.dt.month
    df['quarter'] = d.dt.quarter
    df['day'] = d.dt.day
    df['day_of_week'] = d.dt.dayofweek
    df['day_of_year'] = d.dt.dayofyear
    df['week_of_year'] = d.dt.isocalendar().week.astype(int)
    df['is_weekend'] = (d.dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = d.dt.is_month_start.astype(int)
    df['is_month_end'] = d.dt.is_month_end.astype(int)
    df['is_quarter_start'] = d.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = d.dt.is_quarter_end.astype(int)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['day_of_week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    df['week_of_year_sin'] = np.sin(2*np.pi*df['week_of_year']/52)
    df['week_of_year_cos'] = np.cos(2*np.pi*df['week_of_year']/52)
    df['day_of_year_sin'] = np.sin(2*np.pi*df['day_of_year']/365.25)
    df['day_of_year_cos'] = np.cos(2*np.pi*df['day_of_year']/365.25)
    return df


def add_demand_history_features(df):
    df = df.copy()
    for lag in DEMAND_LAGS:
        df[f'demand_lag_{lag}'] = _shift(df, TARGET_COLUMN, lag)
    for w in ROLLING_WINDOWS:
        df[f'demand_rolling_mean_{w}'] = _rolling(df, TARGET_COLUMN, w, 'mean')
        df[f'demand_rolling_std_{w}'] = _rolling(df, TARGET_COLUMN, w, 'std')
        df[f'demand_rolling_min_{w}'] = _rolling(df, TARGET_COLUMN, w, 'min')
        df[f'demand_rolling_max_{w}'] = _rolling(df, TARGET_COLUMN, w, 'max')
    return df


def add_trend_features(df):
    df = df.copy()
    m7, m14, m28 = df['demand_rolling_mean_7'], df['demand_rolling_mean_14'], df['demand_rolling_mean_28']
    m56, m90 = df['demand_rolling_mean_56'], df['demand_rolling_mean_90']
    df['demand_momentum_7_28'] = _safe_div(m7, m28, 1.0)
    df['demand_momentum_7_56'] = _safe_div(m7, m56, 1.0)
    df['demand_momentum_7_90'] = _safe_div(m7, m90, 1.0)
    df['demand_momentum_14_28'] = _safe_div(m14, m28, 1.0)
    df['demand_momentum_28_90'] = _safe_div(m28, m90, 1.0)
    df['demand_trend_7_28'] = m7 - m28
    df['demand_trend_7_56'] = m7 - m56
    df['demand_trend_7_90'] = m7 - m90
    df['demand_volatility_7_28'] = _safe_div(df['demand_rolling_std_7'], df['demand_rolling_std_28'], 1.0)
    df['demand_volatility_14_56'] = _safe_div(df['demand_rolling_std_14'], df['demand_rolling_std_56'], 1.0)
    return df


def add_historical_group_features(df):
    df = df.copy()
    prev = _shift(df, TARGET_COLUMN, 1)
    temp = df.assign(_previous_demand=prev)
    sg = temp.groupby('Store ID', sort=False)['_previous_demand']
    pg = temp.groupby('Product ID', sort=False)['_previous_demand']
    xg = temp.groupby(GROUP_KEYS, sort=False)['_previous_demand']
    df['store_historical_avg_demand'] = sg.transform(lambda x: x.expanding(min_periods=1).mean())
    df['store_historical_std_demand'] = sg.transform(lambda x: x.expanding(min_periods=2).std())
    df['product_historical_avg_demand'] = pg.transform(lambda x: x.expanding(min_periods=1).mean())
    df['product_historical_std_demand'] = pg.transform(lambda x: x.expanding(min_periods=2).std())
    df['series_historical_avg_demand'] = xg.transform(lambda x: x.expanding(min_periods=1).mean())
    df['series_historical_std_demand'] = xg.transform(lambda x: x.expanding(min_periods=2).std())
    df['recent_vs_store_avg'] = _safe_div(df['demand_rolling_mean_7'], df['store_historical_avg_demand'], 1.0)
    df['recent_vs_product_avg'] = _safe_div(df['demand_rolling_mean_7'], df['product_historical_avg_demand'], 1.0)
    df['recent_vs_series_avg'] = _safe_div(df['demand_rolling_mean_7'], df['series_historical_avg_demand'], 1.0)
    return df


def add_inventory_features(df):
    df = df.copy()
    ref = df['demand_rolling_mean_7']
    df['inventory_to_demand_ratio'] = _safe_div(df['Inventory Level'], ref)
    df['order_to_demand_ratio'] = _safe_div(df['Units Ordered'], ref)
    df['sales_to_demand_ratio'] = _safe_div(df['Units Sold'], ref)
    df['inventory_minus_recent_demand'] = df['Inventory Level'] - ref
    df['order_minus_recent_demand'] = df['Units Ordered'] - ref
    return df


def build_features(df, drop_initial_missing=False):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(GROUP_KEYS + ['Date']).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_demand_history_features(df)
    df = add_trend_features(df)
    df = add_historical_group_features(df)
    df = add_inventory_features(df)
    if drop_initial_missing:
        df = df.dropna(subset=['demand_lag_90', 'demand_rolling_mean_90']).reset_index(drop=True)
    return df


def get_feature_columns():
    return [
        'Store ID','Product ID','Category','Region',
        'year','month','quarter','day','day_of_week','day_of_year','week_of_year',
        'is_weekend','is_month_start','is_month_end','is_quarter_start','is_quarter_end',
        'month_sin','month_cos','day_of_week_sin','day_of_week_cos',
        'week_of_year_sin','week_of_year_cos','day_of_year_sin','day_of_year_cos',
        *[f'demand_lag_{x}' for x in DEMAND_LAGS],
        *[f'demand_rolling_mean_{w}' for w in ROLLING_WINDOWS],
        *[f'demand_rolling_std_{w}' for w in ROLLING_WINDOWS],
        *[f'demand_rolling_min_{w}' for w in ROLLING_WINDOWS],
        *[f'demand_rolling_max_{w}' for w in ROLLING_WINDOWS],
        'demand_momentum_7_28','demand_momentum_7_56','demand_momentum_7_90',
        'demand_momentum_14_28','demand_momentum_28_90',
        'demand_trend_7_28','demand_trend_7_56','demand_trend_7_90',
        'demand_volatility_7_28','demand_volatility_14_56',
        'store_historical_avg_demand','store_historical_std_demand',
        'product_historical_avg_demand','product_historical_std_demand',
        'series_historical_avg_demand','series_historical_std_demand',
        'recent_vs_store_avg','recent_vs_product_avg','recent_vs_series_avg',
        'Inventory Level','Units Ordered','Units Sold','Price','Discount',
        'Promotion','Weather Condition','Seasonality','Epidemic','Competitor Pricing',
        'inventory_to_demand_ratio','order_to_demand_ratio','sales_to_demand_ratio',
        'inventory_minus_recent_demand','order_minus_recent_demand'
    ]


def prepare_training_data(path=None):
    raw = load_data(path)
    feature_df = build_features(raw, drop_initial_missing=True)
    features = get_feature_columns()
    missing = [c for c in features if c not in feature_df.columns]
    if missing:
        raise ValueError('Missing engineered features: ' + ', '.join(missing))
    return feature_df[features].copy(), feature_df[TARGET_COLUMN].copy(), feature_df


def validate_features(path=None):
    raw = load_data(path)
    engineered = build_features(raw, drop_initial_missing=False)
    features = get_feature_columns()
    return {
        'raw_shape': tuple(raw.shape),
        'feature_shape': tuple(engineered.shape),
        'date_min': str(raw['Date'].min().date()),
        'date_max': str(raw['Date'].max().date()),
        'stores': int(raw['Store ID'].nunique()),
        'products': int(raw['Product ID'].nunique()),
        'store_product_series': int(raw.groupby(GROUP_KEYS).ngroups),
        'missing_values_raw': int(raw.isna().sum().sum()),
        'model_feature_count': len(features),
        'missing_model_features': [c for c in features if c not in engineered.columns],
    }


if __name__ == '__main__':
    print('=' * 70)
    print('PHASE 1 - ADVANCED FEATURE ENGINEERING VALIDATION')
    print('=' * 70)
    d = validate_features()
    for k, v in d.items():
        print(f'{k}: {v}')
    if d['missing_model_features']:
        raise SystemExit('Feature validation failed.')
    X, y, f = prepare_training_data()
    print('-' * 70)
    print(f'Training matrix shape: {X.shape}')
    print(f'Target shape: {y.shape}')
    print(f'Number of model features: {len(get_feature_columns())}')
    print('-' * 70)
    print(f[['demand_lag_90','demand_rolling_mean_28','demand_rolling_mean_90',
          'demand_momentum_7_28','demand_trend_7_90','series_historical_avg_demand',
          'recent_vs_series_avg']].head().to_string(index=False))
    print('=' * 70)
    print('PHASE 1 PASSED')
    print('=' * 70)

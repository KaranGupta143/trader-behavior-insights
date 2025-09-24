# Trader Behavior Insights Project

"""
Usage:
- Open this file in JupyterLab/Colab and run cells sequentially.
- Replace the data paths with local CSV paths or mount Google Drive.

Deliverables inside this notebook:
- Data loading & validation
- Exploratory Data Analysis (EDA)
- Data cleaning & feature engineering
- Sentiment alignment (Fear & Greed Index) to trades
- Aggregate trader performance metrics
- Statistical tests and visualizations
- Baseline predictive model(s) for trader profitability
- Exportable artifacts: CSV summaries, charts, model files

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost (optional), statsmodels

"""

# %%
# 1) Environment & imports
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# statistics & modeling
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score


# optional: XGBoost if installed
try:
    import xgboost as xgb
    _xgb_available = True
except Exception:
    _xgb_available = False
# 2) Paths and helper functions
# Replace these with the correct file locations or mount Google Drive in Colab
HISTORICAL_DATA_PATH = 'historical_data.csv'  # <- replace or upload file
FEAR_GREED_PATH = 'fear_greed_index.csv'     # <- replace or upload file
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv_try(paths):
    """Try multiple paths until file loads. Useful when running locally vs Colab."""
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                print(f"Loaded {p} with shape {df.shape}")
                return df
            except Exception as e:
                print(f"Error loading {p}: {e}")
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


# 3) Load datasets (replace paths if needed)
try:
    trades = load_csv_try([HISTORICAL_DATA_PATH])
    fg = load_csv_try([FEAR_GREED_PATH])
except FileNotFoundError as e:
    print("Data files not found.\n- If using Google Drive links, make them publicly viewable or download and set HISTORICAL_DATA_PATH and FEAR_GREED_PATH to local filenames.\n- Alternatively, upload CSVs to the notebook environment and re-run.")
    raise

# 4) Quick preview
print('\n--- Trades ---')
print(trades.head().T)
print('\n--- Fear & Greed ---')
print(fg.head().T)

# 5) Standardize time columns
# Common problems: epoch ms, timezone, missing tz info, inconsistent formats

# Example heuristics (adjust column names based on dataset):
# trades: columns 'time' or 'timestamp' or 'date'
# fg: columns 'Date' and 'Classification' expected

# Detect trade time column
time_cols = [c for c in trades.columns if 'time' in c.lower() or 'date' in c.lower() or 'timestamp' in c.lower()]
print('Candidate time columns in trades:', time_cols)

# Use heuristics to parse
trade_time_col = time_cols[0] if time_cols else None
if trade_time_col is None:
    raise ValueError('No time-like column found in trades. Please inspect column names.')

# parse to datetime
trades['trade_time'] = pd.to_datetime(trades[trade_time_col], unit='ms', errors='coerce')
# fallback to parse without unit
mask = trades['trade_time'].isna()
if mask.any():
    trades.loc[mask, 'trade_time'] = pd.to_datetime(trades.loc[mask, trade_time_col], errors='coerce')

# For fear_greed dataset: expect a 'Date' column
if 'Date' in fg.columns:
    fg['fg_date'] = pd.to_datetime(fg['Date'], errors='coerce')
else:
    # try other names
    date_cols = [c for c in fg.columns if 'date' in c.lower()]
    if date_cols:
        fg['fg_date'] = pd.to_datetime(fg[date_cols[0]], errors='coerce')
    else:
        raise ValueError('No date column found in fear/greed dataset.')

# normalize date to date (no time) for joining daily sentiment
fg['fg_date'] = fg['fg_date'].dt.normalize()
trades['trade_date'] = trades['trade_time'].dt.normalize()

# 6) Clean & canonicalize columns
# Example canonical columns we will use: account, symbol, executionPrice, size, side, trade_time, trade_date, closedPnL, leverage

# lowercase columns for safety
trades.columns = [c.strip() for c in trades.columns]

# rename common variants if present
rename_map = {}
for candidate in ['execution price', 'execution_price', 'price', 'execPrice']:
    for c in trades.columns:
        if c.lower().replace(' ', '') == candidate.replace(' ', ''):
            rename_map[c] = 'execution_price'
for candidate in ['closedpnl', 'closedPnL', 'closed_pnl', 'pnl']:
    for c in trades.columns:
        if c.lower().replace(' ', '') == candidate.lower():
            rename_map[c] = 'closedPnL'

if rename_map:
    trades = trades.rename(columns=rename_map)
    print('Renamed columns:', rename_map)

# ensure numeric columns are numeric
for col in ['execution_price', 'size', 'closedPnL', 'leverage']:
    if col in trades.columns:
        trades[col] = pd.to_numeric(trades[col], errors='coerce')

# side normalization
if 'side' in trades.columns:
    trades['side'] = trades['side'].astype(str).str.lower().str.strip()
    trades['side'] = trades['side'].replace({'buy':'long','sell':'short','b':'long','s':'short'})




# symbol cleanup
if 'symbol' in trades.columns:
    trades['symbol'] = trades['symbol'].astype(str).str.upper().str.strip()

# 7) Merge sentiment into trades on date
# If Fear & Greed has numeric index score, also include it; else map categorical labels

# Preview fg columns
print('\nFear & Greed columns:', fg.columns.tolist())

# Attempt to find numeric score column
score_cols = [c for c in fg.columns if 'score' in c.lower() or 'value' in c.lower() or 'index' in c.lower()]
label_cols = [c for c in fg.columns if 'class' in c.lower() or 'classification' in c.lower() or 'label' in c.lower()]

fg_lookup = fg[['fg_date'] + score_cols + label_cols].copy()
# if no label cols, guess 'Classification'
if not label_cols and 'Classification' in fg.columns:
    fg_lookup['classification'] = fg['Classification']

# rename to consistent names
if score_cols:
    fg_lookup = fg_lookup.rename(columns={score_cols[0]:'fg_score'})
if label_cols or 'classification' in fg_lookup.columns:
    lab = label_cols[0] if label_cols else 'classification'
    fg_lookup = fg_lookup.rename(columns={lab:'fg_label'})

# merge
trades = trades.merge(fg_lookup, left_on='trade_date', right_on='fg_date', how='left')

# If sentiment missing for some dates, optionally forward/backfill
trades['fg_score'] = trades.get('fg_score')
trades['fg_label'] = trades.get('fg_label')#8 Aggregate per Account and per Day
trades['is_profitable'] = trades['closedPnL'] > 0

agg_funcs = {
    'closedPnL': ['sum', 'mean', 'std'],
    'is_profitable': ['mean'],
    'execution_price': ['count']
}

account_daily = trades.groupby(['Account', 'trade_date']).agg(agg_funcs)
account_daily.columns = ['_'.join(col).strip() for col in account_daily.columns.values]
account_daily = account_daily.reset_index()

# Join sentiment back
account_daily = account_daily.merge(fg_lookup, left_on='trade_date', right_on='fg_date', how='left')

# Save output CSV
account_daily.to_csv(os.path.join(OUTPUT_DIR, 'account_daily_summary.csv'), index=False)
# 9) Exploratory plots

def simple_bar(x, y, title):
    plt.figure(figsize=(8,4))
    sns.barplot(x=x, y=y, data=account_daily)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example: average profit by fg_label
if 'fg_label' in account_daily.columns:
    tmp = account_daily.groupby('fg_label')['closedPnL_sum'].mean().reset_index()
    plt.figure(figsize=(8,4))
    sns.barplot(x='fg_label', y='closedPnL_sum', data=tmp)
    plt.title('Avg daily PnL by Fear/Greed label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'avg_pnl_by_fg_label.png'))
    plt.show()

# Profitability rate vs sentiment score scatter
if 'fg_score' in account_daily.columns:
    plt.figure(figsize=(8,4))
    sns.scatterplot(x='fg_score', y='is_profitable_mean', data=account_daily, alpha=0.6)
    plt.title('Profitability rate vs FG score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'profit_rate_vs_fg_score.png'))
    plt.show()
# 10) Statistical testing
# Does average daily PnL differ between Fear vs Greed days?
if 'fg_label' in account_daily.columns and 'closedPnL_sum' in account_daily.columns:
    labels = account_daily['fg_label'].dropna().unique()
    print('FG labels:', labels)
    if len(labels) >= 2:
        groups = [account_daily.loc[account_daily['fg_label']==lab, 'closedPnL_sum'].dropna() for lab in labels]
        # pairwise t-test (example for first two labels)
        tstat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        print(f'T-test between {labels[0]} and {labels[1]}: t={tstat:.3f}, p={pval:.3e}')

# 11) Build a baseline predictive model (classification: whether a trade will be profitable)
# Features: symbol-level features, lagged sentiment, leverage, size, entry price, time-of-day, historical win-rate

# Prepare trade-level features
feature_df = trades.copy()
# example features
for col in ['size','leverage','execution_price']:
    if col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

# time features
feature_df['hour'] = feature_df['trade_time'].dt.hour
feature_df['weekday'] = feature_df['trade_time'].dt.weekday

# encode side
feature_df['side_encoded'] = (feature_df['side']=='long').astype(int) if 'side' in feature_df.columns else 0

# label
feature_df['target_profitable'] = (feature_df['closedPnL']>0).astype(int)

# simple feature selection
feat_cols = [c for c in ['size','leverage','execution_price','hour','weekday','side_encoded','fg_score'] if c in feature_df.columns]
feature_df = feature_df.dropna(subset=['target_profitable'])
X = feature_df[feat_cols].fillna(0)
y = feature_df['target_profitable']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# feature importances
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\nFeature importances:\n', importances)
importances.to_csv(os.path.join(OUTPUT_DIR, 'feature_importances.csv'))

# 12) Save model (optional)
try:
    import joblib
    joblib.dump(clf, os.path.join(OUTPUT_DIR, 'rf_profitable_model.pkl'))
    print('Saved model to outputs/rf_profitable_model.pkl')
except Exception as e:
    print('joblib not available or failed to save model:', e)

# 13)Aggregate per account per day
agg_funcs = {
    'closedPnL': ['sum', 'mean', 'std'],
    'is_profitable': ['mean'],
    'execution_price': ['count']
}

account_daily = trades.groupby(['Account', 'trade_date']).agg(agg_funcs)
account_daily.columns = ['_'.join(col).strip() for col in account_daily.columns.values]
account_daily = account_daily.reset_index()

# Merge with sentiment
account_daily = account_daily.merge(fg_lookup, left_on='trade_date', right_on='fg_date', how='left')
account_daily.to_csv(os.path.join(OUTPUT_DIR, 'account_daily_summary.csv'), index=False)

# Per-account Sharpe ratio
acct_pnl = trades.groupby('Account')['closedPnL'].agg(['mean','std','sum'])
acct_pnl['sharpe'] = acct_pnl['mean'] / (acct_pnl['std'] + 1e-9) * np.sqrt(252)
acct_pnl.to_csv(os.path.join(OUTPUT_DIR, 'account_pnl_stats.csv'))
# 14) Final deliverables & next steps
print('\nDeliverables written to outputs/:')
for f in os.listdir(OUTPUT_DIR):
    print(' -', f)

print('\nRecommended next steps:')
print('1) Run this notebook with the provided CSVs.\n2) Inspect saved outputs and charts.\n3) Iterate on feature engineering: symbol momentum, intra-day volatility, prior-win streaks.\n4) If needed, build time-series models at aggregate level (ARIMA/Prophet) or causal inference experiments.')

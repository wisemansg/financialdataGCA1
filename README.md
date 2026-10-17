"""
# FD-GCA: Financial Data — Graded Code Activity 1

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/Financial%20Data-Analysis-blue" />
  <img src="https://img.shields.io/badge/Data%20Engineering-Preprocessing-orange" />
</p>

---

## Overview
This repository documents Graded Code Activity 1 for Financial Data Processing & Analytics. The activity covers end-to-end data pipelines for financial datasets, including multi-asset market data ingestion, missing value imputation, log-return processing, rolling volatility metrics, risk-adjusted performance statistics (Sharpe and Sortino ratios), and empirical distribution diagnostics.

---

## Part 1: Market Data Ingestion & Imputation

### Problem 1.1: Data Alignment and Cleaning

#### Objective
Clean multi-asset financial price series by handling missing values ($\text{NaN}$s) using forward-fill ($\text{ffill}$) followed by backward-fill ($\text{bfill}$) to preserve non-synchronous asset pricing structures.

#### Mathematical Definition
For price series $P_{i,t}$, missing value imputation follows:
$$P_{i,t}^{\text{imputed}} = \begin{cases} P_{i,t}, & \text{if } P_{i,t} \text{ exists} \\ P_{i,t-\tau}, & \text{where } \tau = \min\{k > 0 : P_{i,t-k} \text{ exists}\} \end{cases}$$

---

### Problem 1.2: Log-Return Matrix Computation

#### Objective
Construct the asset continuous log-return matrix $\mathbf{R} \in \mathbb{R}^{T \times N}$ from imputed asset price levels $P_{i,t}$.

#### Mathematical Formula
$$r_{i,t} = \ln(P_{i,t}) - \ln(P_{i,t-1}) = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

---

## Part 2: Risk-Adjusted Performance Diagnostics

### Problem 2.1: Sharpe & Sortino Ratios

#### Objective
Compute annualized Sharpe Ratio ($\text{SR}$) and Sortino Ratio ($\text{SoR}$) for an asset assuming $252$ trading days per year and a risk-free rate $r_f$.

#### Mathematical Formulas
Annualized Mean Return and Volatility:
$$\mu_{\text{ann}} = 252 \cdot \bar{r}, \quad \sigma_{\text{ann}} = \sqrt{252} \cdot \sigma_r$$

Sharpe Ratio:
$$\text{SR} = \frac{\mu_{\text{ann}} - r_f}{\sigma_{\text{ann}}}$$

Downside Risk Deviation ($\sigma_d$) and Sortino Ratio:
$$\sigma_{d,\text{ann}} = \sqrt{252} \cdot \sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, r_t - r_{f,\text{daily}})^2}$$

$$\text{SoR} = \frac{\mu_{\text{ann}} - r_f}{\sigma_{d,\text{ann}}}$$

---

## Part 3: Rolling Volatility & Empirical Risk Diagnostics

### Problem 3.1: Rolling Volatility and Maximum Drawdown

#### Objective
Compute a $W$-day rolling annualized volatility series $\sigma_{t,W}$ and evaluate historical Maximum Drawdown ($\text{MDD}$) across the price trajectory.

#### Mathematical Formulas
Rolling Annualized Volatility:
$$\sigma_{t,W} = \sqrt{252} \cdot \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} (r_{t-k} - \bar{r}_{t,W})^2}$$

Peak-to-Trough Maximum Drawdown ($\text{MDD}$):
$$\text{MDD} = \max_{t} \left( \frac{\max_{\tau \le t} P_\tau - P_t}{\max_{\tau \le t} P_\tau} \right)$$

---

## Summary Table of Outputs & Validations

| Task ID | Output Parameter | Metric / Description | Theoretical Target / Formula |
|:---|:---|:---|:---|
| P1.1 | `clean_prices` | Forward-filled and backward-filled prices | Zero missing values ($\text{NaN} = 0$) |
| P1.2 | `log_returns` | Asset log-returns matrix $r_{i,t}$ | $\ln(P_{i,t} / P_{i,t-1})$ |
| P2.1 | `sharpe_ratio` | Annualized risk-adjusted excess return | $\frac{\mu_{\text{ann}} - r_f}{\sigma_{\text{ann}}}$ |
| P2.1 | `sortino_ratio` | Downside-adjusted excess return | $\frac{\mu_{\text{ann}} - r_f}{\sigma_{d,\text{ann}}}$ |
| P3.1 | `max_drawdown` | Peak-to-trough maximum loss percentage | $\max_{t} \left( \frac{\max_{\tau \le t} P_\tau - P_t}{\max_{\tau \le t} P_\tau} \right)$ |
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ==========================================
# PART 1: DATA INGESTION & PREPROCESSING
# ==========================================

def preprocess_market_data(raw_prices: pd.DataFrame) -> dict:
    """
    Cleans multi-asset market data by imputing missing values and computing continuous log-returns.

    Parameters
    ----------
    raw_prices : pd.DataFrame
        DataFrame of asset closing prices with potential missing values.

    Returns
    -------
    dict
        Contains cleaned price DataFrame and computed log-returns DataFrame.
    """
    # 1. Forward-fill then backward-fill missing price observations
    clean_prices = raw_prices.ffill().bfill()

    # 2. Compute continuous log-returns matrix: r_{i,t} = \ln(P_{i,t} / P_{i,t-1})
    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()

    return {
        'clean_prices': clean_prices,
        'log_returns': log_returns
    }


# ==========================================
# PART 2: RISK-ADJUSTED METRICS
# ==========================================

def compute_risk_adjusted_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """
    Calculates annualized Sharpe Ratio and Sortino Ratio for a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns series $r_t$.
    risk_free_rate : float
        Annualized risk-free benchmark rate $r_f$ (default = 0.02).

    Returns
    -------
    dict
        Annualized mean, volatility, Sharpe Ratio, and Sortino Ratio.
    """
    rf_daily = risk_free_rate / 252.0

    # 1. Annualized mean and total volatility
    mean_daily = returns.mean()
    std_daily = returns.std(ddof=1)
    
    mu_ann = mean_daily * 252.0
    sigma_ann = std_daily * np.sqrt(252.0)

    # 2. Sharpe Ratio: \text{SR} = \frac{\mu_{\text{ann}} - r_f}{\sigma_{\text{ann}}}
    sharpe_ratio = (mu_ann - risk_free_rate) / sigma_ann if sigma_ann > 0 else np.nan

    # 3. Downside risk deviation and Sortino Ratio
    downside_diff = np.minimum(0.0, returns - rf_daily)
    downside_std_daily = np.sqrt(np.mean(downside_diff**2))
    sigma_d_ann = downside_std_daily * np.sqrt(252.0)

    sortino_ratio = (mu_ann - risk_free_rate) / sigma_d_ann if sigma_d_ann > 0 else np.nan

    return {
        'mu_annualized': float(np.round(mu_ann, 6)),
        'sigma_annualized': float(np.round(sigma_ann, 6)),
        'sharpe_ratio': float(np.round(sharpe_ratio, 4)),
        'sortino_ratio': float(np.round(sortino_ratio, 4))
    }


# ==========================================
# PART 3: ROLLING VOLATILITY & DRAWDOWN
# ==========================================

def compute_rolling_and_drawdown(prices: pd.Series, returns: pd.Series, window: int = 21) -> dict:
    """
    Computes rolling annualized volatility and maximum peak-to-trough drawdown.

    Parameters
    ----------
    prices : pd.Series
        Daily asset price trajectory $P_t$.
    returns : pd.Series
        Daily log-returns series $r_t$.
    window : int
        Rolling window size in trading days $W$ (default = 21).

    Returns
    -------
    dict
        Rolling volatility series, latest rolling volatility, and Maximum Drawdown.
    """
    # 1. Rolling annualized volatility: \sigma_{t,W} = \sqrt{252} \cdot \sigma_{\text{rolling}}
    rolling_vol_ann = returns.rolling(window=window).std(ddof=1) * np.sqrt(252.0)

    # 2. Maximum Drawdown calculation: \text{MDD} = \max_t \left( \frac{\max_{\tau \le t} P_\tau - P_t}{\max_{\tau \le t} P_\tau} \right)
    cumulative_max = prices.cummax()
    drawdown = (cumulative_max - prices) / cumulative_max
    max_drawdown = float(drawdown.max())

    return {
        'rolling_volatility_series': rolling_vol_ann,
        'latest_rolling_volatility': float(np.round(rolling_vol_ann.iloc[-1], 6)),
        'max_drawdown': float(np.round(max_drawdown, 6))
    }


# ==========================================
# EXECUTION TEST & VALIDATION RUNNER
# ==========================================

if __name__ == "__main__":
    print("Executing Financial Data Graded Code Activity 1 Pipeline...\n")

    # Generate synthetic multi-asset price DataFrame with missing values
    np.random.seed(42)
    n_obs = 500
    dates = pd.date_range(start="2024-01-01", periods=n_obs, freq="B")
    
    price_a = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.012, n_obs)))
    price_b = 50.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, n_obs)))
    
    df_raw = pd.DataFrame({'Asset_A': price_a, 'Asset_B': price_b}, index=dates)
    
    # Introduce random missing values (NaNs)
    df_raw.iloc[5:8, 0] = np.nan
    df_raw.iloc[20:22, 1] = np.nan

    # Test Part 1: Data Preprocessing
    prep_res = preprocess_market_data(df_raw)
    print("--- Part 1: Preprocessing & Log-Returns ---")
    print(f"Cleaned Prices Shape : {prep_res['clean_prices'].shape}")
    print(f"Log Returns Shape    : {prep_res['log_returns'].shape}")

    # Test Part 2: Risk-Adjusted Metrics for Asset A
    metrics_a = compute_risk_adjusted_metrics(prep_res['log_returns']['Asset_A'], risk_free_rate=0.02)
    print("\n--- Part 2: Performance Metrics (Asset A) ---")
    print(f"Annualized Mean Return : {metrics_a['mu_annualized']}")
    print(f"Annualized Volatility  : {metrics_a['sigma_annualized']}")
    print(f"Sharpe Ratio           : {metrics_a['sharpe_ratio']}")
    print(f"Sortino Ratio          : {metrics_a['sortino_ratio']}")

    # Test Part 3: Rolling Volatility & Drawdown
    drawdown_res = compute_rolling_and_drawdown(
        prep_res['clean_prices']['Asset_A'], 
        prep_res['log_returns']['Asset_A'], 
        window=21
    )
    print("\n--- Part 3: Volatility & Drawdown Diagnostics ---")
    print(f"Latest 21-Day Volatility : {drawdown_res['latest_rolling_volatility']}")
    print(f"Maximum Drawdown         : {drawdown_res['max_drawdown']}")

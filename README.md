
# FE610-GCA: Financial Econometrics — Graded Code Activity 1

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/Statsmodels-012169?logo=statsmodels&logoColor=white" />
  <img src="https://img.shields.io/badge/ARCH%20Model-008080?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Time%20Series%20Analysis-FF6F00?logo=analytics&logoColor=white" />
  <img src="https://img.shields.io/badge/Financial%20Econometrics-MScFE%20610-blue" />
</p>

---

## Overview
This repository documents Graded Code Activity 1 for MScFE 610: Financial Econometrics. The activity covers foundational econometrics workflows, including return calculation, asset return properties, stationarity testing, and baseline time-series estimation ($AR$, $MA$, $ARIMA$, and $ARCH$).

---

## Part 1: Financial Time Series Preprocessing & Properties

### Problem 1.1: Log-Returns and Summary Statistics

#### Objective
Compute daily log-returns from raw asset closing prices and calculate baseline descriptive statistics: Mean ($\mu$), Standard Deviation ($\sigma$), Skewness ($S$), and Excess Kurtosis ($K$).

#### Mathematical Definition
Given closing price $P_t$, the continuous log-return $r_t$ is defined as:
$$r_t = \ln(P_t) - \ln(P_{t-1}) = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

---

### Problem 1.2: Augmented Dickey-Fuller (ADF) Stationarity Diagnostic

#### Objective
Test whether closing prices $P_t$ and continuous log-returns $r_t$ are stationary processes using the Augmented Dickey-Fuller test.

#### Regression Model
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t$$

#### Hypotheses
- $H_0: \gamma = 0$ (Unit root present; non-stationary process).  
- $H_1: \gamma < 0$ (No unit root; stationary process).

---

## Part 2: ARMA / ARIMA Parameter Estimation

### Problem 2.1: Order Identification via ACF/PACF & Automated Search

#### Objective
Fit an $ARIMA(p, d, q)$ model to daily log-returns, selecting optimal orders $(p, q)$ by minimizing the Akaike Information Criterion ($\text{AIC}$).

---

## Part 3: Baseline ARCH(1) Model Estimation

### Problem 3.1: Volatility Clustering & ARCH(1) Specification

#### Objective
Estimate a zero-mean $ARCH(1)$ volatility model to model conditional heteroskedasticity in daily log-returns $r_t$:
$$r_t = \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$
$$\sigma_t^2 = \omega + \alpha_1 r_{t-1}^2$$

Calculate parameter estimates $\hat{\omega}$ and $\hat{\alpha}_1$, persistence $\alpha_1$, and one-step-ahead variance forecast ratio $\frac{\sigma_{t+1}^2}{\bar{\sigma}^2}$.

#### Mathematical Formulas
- **Persistence:** $\alpha_1$  
- **Unconditional Variance:** $\sigma^2 = \frac{\omega}{1 - \alpha_1}$  
- **One-Step-Ahead Variance Forecast Ratio:** 
$$\frac{\sigma_{t+1}^2}{\bar{\sigma}^2} = \frac{\omega + \alpha_1 r_t^2}{\text{Var}(r)}$$

---

## Summary Table of Outputs & Validations

| Task ID | Output Parameter | Metric / Description | Theoretical Target / Formula |
|:---|:---|:---|:---|
| P1.1 | $\mu, \sigma, S, K$ | Log-return distribution moments | Empirical calculations on continuous returns $r_t$ |
| P1.2 | $p\text{-value}$ | ADF test $p$-value | $p < 0.05 \implies$ Rejection of unit root $H_0$ |
| P2.1 | $(p, d, q)$ | Model selection $(p, d, q)$ | $\min \text{AIC}(p, q)$ |
| P3.1 | $\frac{\sigma_{T+1}^2}{\text{Var}(r)}$ | Volatility shock forecast impact | $\frac{\omega + \alpha_1 r_T^2}{\text{Var}(r)}$ |
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ==========================================
# PART 1: PREPROCESSING & STATIONARITY
# ==========================================

def compute_return_statistics(prices: pd.Series) -> dict:
    """
    Computes log-returns $r_t = \ln(P_t) - \ln(P_{t-1})$ and baseline descriptive 
    statistics: Mean ($\mu$), Standard Deviation ($\sigma$), Skewness ($S$), 
    and Excess Kurtosis ($K$).

    Parameters
    ----------
    prices : pd.Series
        Series of daily closing prices $P_t$.

    Returns
    -------
    dict
        Summary statistics rounded to 6 decimal places.
    """
    # 1. Compute log returns: $r_t = \ln(P_t) - \ln(P_{t-1})$
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # 2. Summary statistics: $\mu, \sigma, S, K$
    mean_ret = log_returns.mean()
    std_ret = log_returns.std(ddof=1)
    skew_ret = skew(log_returns)
    kurt_ret = kurtosis(log_returns, fisher=True)  # Excess kurtosis ($K = 0$ for Normal distribution)

    return {
        'mean': float(np.round(mean_ret, 6)),
        'std': float(np.round(std_ret, 6)),
        'skewness': float(np.round(skew_ret, 6)),
        'excess_kurtosis': float(np.round(kurt_ret, 6)),
        'log_returns': log_returns
    }


def run_adf_diagnostics(prices: pd.Series, log_returns: pd.Series) -> dict:
    """
    Performs ADF unit root tests on price levels $P_t$ and log-returns $r_t$.
    
    Tests $H_0: \gamma = 0$ against $H_1: \gamma < 0$ in the regression model:
    $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t$$
    """
    # ADF test on raw price level $P_t$
    adf_price_stat, adf_price_p, _, _, _, _ = adfuller(prices.dropna(), autolag='AIC')

    # ADF test on log-returns $r_t$
    adf_ret_stat, adf_ret_p, _, _, _, _ = adfuller(log_returns.dropna(), autolag='AIC')

    return {
        'price_adf_stat': float(np.round(adf_price_stat, 4)),
        'price_pvalue': float(np.round(adf_price_p, 4)),
        'returns_adf_stat': float(np.round(adf_ret_stat, 4)),
        'returns_pvalue': float(np.round(adf_ret_p, 4)),
        'returns_is_stationary': bool(adf_ret_p < 0.05)
    }


# ==========================================
# PART 2: ARIMA MODEL ESTIMATION
# ==========================================

def fit_optimal_arima(log_returns: pd.Series, max_p: int = 3, max_q: int = 3) -> dict:
    """
    Performs grid search to identify optimal $ARIMA(p, d, q)$ specification 
    by minimizing the Akaike Information Criterion ($\text{AIC}$).
    """
    best_aic = float('inf')
    best_order = (0, 0, 0)
    best_model_res = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(log_returns, order=(p, 0, q))
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, 0, q)
                    best_model_res = res
            except Exception:
                continue

    return {
        'optimal_order': best_order,
        'aic': float(np.round(best_aic, 4)),
        'params': best_model_res.params.round(6).to_dict() if best_model_res is not None else {}
    }


# ==========================================
# PART 3: BASELINE ARCH(1) ESTIMATION
# ==========================================

def fit_arch1_model(log_returns: pd.Series) -> dict:
    """
    Fits an $ARCH(1)$ model with Normal innovations $\epsilon_t \sim \mathcal{N}(0, 1)$ to log-returns $r_t$:
    $$r_t = \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$$
    $$\sigma_t^2 = \omega + \alpha_1 r_{t-1}^2$$

    Parameters
    ----------
    log_returns : pd.Series
        Daily log-returns array/series $r_t$.

    Returns
    -------
    dict
        Contains parameter estimates $\hat{\omega}$, $\hat{\alpha}_1$, 
        persistence $\alpha_1$, unconditional variance $\sigma^2 = \frac{\omega}{1-\alpha_1}$, 
        and variance ratio $\frac{\sigma_{T+1}^2}{\text{Var}(r)}$.
    """
    # 1. Initialize and fit ARCH(1) zero-mean model
    am = arch_model(log_returns, vol='ARCH', p=1, q=0, mean='Zero', dist='Normal', rescale=False)
    res = am.fit(disp='off')

    # 2. Extract parameters $\hat{\omega}$ and $\hat{\alpha}_1$
    omega = float(res.params['omega'])
    alpha1 = float(res.params['alpha[1]'])

    # 3. Calculate derived metrics: Persistence $\alpha_1$, Unconditional Variance $\sigma^2 = \frac{\omega}{1 - \alpha_1}$
    persistence = alpha1
    unconditional_variance = omega / (1.0 - alpha1) if alpha1 < 1.0 else np.nan

    # 4. One-step-ahead forecast for the latest period: $\sigma_{T+1}^2 = \omega + \alpha_1 r_T^2$
    last_return_sq = float(log_returns.iloc[-1] ** 2)
    one_step_var_forecast = omega + alpha1 * last_return_sq
    sample_variance = float(np.var(log_returns, ddof=1))
    variance_ratio = one_step_var_forecast / sample_variance

    return {
        'omega': float(np.round(omega, 6)),
        'alpha1': float(np.round(alpha1, 6)),
        'persistence': float(np.round(persistence, 6)),
        'unconditional_variance': float(np.round(unconditional_variance, 6)),
        'variance_ratio_latest': float(np.round(variance_ratio, 6))
    }


# ==========================================
# EXECUTION TEST & VALIDATION RUNNER
# ==========================================

if __name__ == "__main__":
    print("Executing MScFE 610 Graded Code Activity 1 Pipeline...\n")

    # Generate reproducible synthetic price data
    np.random.seed(42)
    n_days = 500
    price_paths = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_days)))
    prices_series = pd.Series(price_paths, name="Close")

    # Test Part 1.1
    stats_res = compute_return_statistics(prices_series)
    print("--- Part 1.1: Return Statistics ---")
    print(f"Mean: {stats_res['mean']}, Std: {stats_res['std']}, Skew: {stats_res['skewness']}, Excess Kurt: {stats_res['excess_kurtosis']}")

    # Test Part 1.2
    adf_res = run_adf_diagnostics(prices_series, stats_res['log_returns'])
    print("\n--- Part 1.2: ADF Stationarity ---")
    print(f"Returns ADF p-value: {adf_res['returns_pvalue']} | Is Stationary: {adf_res['returns_is_stationary']}")

    # Test Part 2.1
    arima_res = fit_optimal_arima(stats_res['log_returns'], max_p=2, max_q=2)
    print("\n--- Part 2.1: ARIMA Grid Search ---")
    print(f"Optimal Order: {arima_res['optimal_order']} | AIC: {arima_res['aic']}")

    # Test Part 3.1
    arch_res = fit_arch1_model(stats_res['log_returns'])
    print("\n--- Part 3.1: ARCH(1) Estimation ---")
    print(f"Omega: {arch_res['omega']} | Alpha1: {arch_res['alpha1']} | Variance Ratio: {arch_res['variance_ratio_latest']}")

# Market Data Loader & Returns Analysis

**Goal:** Build the foundations of quant trading by learning how to load real market data, compute returns, measure volatility, analyze correlations, and study the distribution of asset returns.

---

## 1. Market Data Collection

**Concept:**  
Financial data is a **time series**: a sequence of prices over time. Quants rarely analyze raw prices directly because what matters is **change**, not absolute level.  

- Raw prices differ across assets (Tesla $200 vs Apple $150).  
- Returns make comparisons meaningful and standardized.  

**Code:**
```python
import yfinance as yf

data = yf.download(["AAPL","MSFT","GOOG","TSLA","SPY"], 
                   start="2018-01-01", end="2025-01-01")["Close"]
data = data.dropna(how="all")
````

---

## 2. Returns vs Log Returns

**Concept:**

* **Simple return (r):**

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

* **Log return (ℓ):**

$$
ℓ_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

Log returns add nicely over time, unlike simple returns.
👉 Think of returns as raw Minecraft XP orbs (messy) vs log returns as your XP bar (clean stacking).

**Code:**

```python
import numpy as np

returns = data.pct_change().dropna()
log_returns = np.log(data / data.shift(1)).dropna()
```

---

## 3. Rolling Statistics

**Concept:**
Markets change over time. Risk isn’t constant.

* **Rolling mean:** recent average return.
* **Rolling volatility:** recent standard deviation (risk).

👉 Like checking the last 30 Minecraft days of mob spawns — was it calm or chaos?

**Code:**

```python
rolling_mean = returns.rolling(30).mean()
rolling_vol = returns.rolling(30).std() * np.sqrt(252)  # annualized
```

---

## 4. Correlation & Correlation Matrix

**Concept:**
Correlation shows how assets move together:

* +1 = perfectly together
* 0 = independent
* –1 = opposite

👉 Portfolio diversification depends on correlations, not just asset count.

**Code:**

```python
corr = returns.corr()

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

---

## 5. Distribution Analysis (Fat Tails, Skewness, Kurtosis)

**Concept:**
Markets are not smooth bell curves. They have:

* **Fat tails:** extreme shocks more frequent than expected
* **Skewness:** asymmetry (e.g., crashes sharper than rallies)
* **Kurtosis:** “spiky” center but heavy tails

👉 Like Minecraft loot drops: sometimes mobs drop something ultra-rare (Elytra), not just bones.

**Code:**

```python
from scipy.stats import skew, kurtosis

vals = returns["AAPL"]
print("Skewness:", skew(vals))
print("Kurtosis:", kurtosis(vals, fisher=True))

plt.hist(vals, bins=50, density=True)
plt.title("Histogram of AAPL Returns")
plt.show()
```

---

## 6. Key Insights to Remember

1. **Returns > Prices:** They standardize changes and reveal real dynamics.
2. **Log Returns:** Preferred for compounding, modeling, and math.
3. **Volatility clusters:** Calm and chaos come in regimes, not randomly.
4. **Correlations matter:** Diversification depends on correlations, not asset count.
5. **Fat tails & skew:** Extreme events are common → risk is underestimated by “normal” models.
6. **This is the foundation:** Every quant model (ARIMA, GARCH, ML, optimization) builds on these basics.

---

## 7. How the Algorithms Work

* **`pct_change()`:** (P\_t - P\_t-1)/P\_t-1 → daily returns.
* **`np.log(data/data.shift(1))`:** log of today vs yesterday → log returns.
* **`.rolling(window).mean()/std()`:** sliding window → rolling stats.
* **`.corr()`:** pairwise Pearson correlations.
* **`skew()/kurtosis()`:** shape of return distribution vs normal curve.

---

## 🎯 One-Sentence Takeaway

Project 1 taught you how to **transform raw price data into returns, measure risk with volatility, study correlations, and understand fat-tailed distributions** — the building blocks of all quantitative trading.

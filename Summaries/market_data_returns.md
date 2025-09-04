# Market Data Loader & Returns Analysis

**Why it matters to quants · Real use-cases · Fully commented Colab code**

**Core goal:** Build the foundation every quant needs — load clean market data, transform prices → returns (and log-returns), measure **volatility**, analyze **correlations**, and understand **return distributions** (fat tails, skewness, kurtosis).

---

## 0) Colab Setup & Structure

**Why quants care**
Reproducibility is professional hygiene. A neat notebook that installs dependencies, fixes a random seed, saves charts to `docs/charts/`, and exports a tidy CSV/PDF makes your work portfolio-ready.

```python
# --- Install dependencies (Colab) ---
!pip -q install yfinance pandas numpy matplotlib seaborn scipy

# --- Imports ---
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import yfinance as yf
from scipy.stats import skew, kurtosis, jarque_bera

# --- Plot aesthetics + output dirs ---
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True
sns.set_style("whitegrid")

os.makedirs("docs/charts", exist_ok=True)
os.makedirs("docs", exist_ok=True)

def save_fig(name):
    """Save figures in a consistent place for your paper/repo."""
    plt.tight_layout()
    plt.savefig(f"docs/charts/{name}.png", dpi=300, bbox_inches="tight")
```

---

## 1) Load Market Data (Yahoo Finance)

**Why quants care**
You’ll do this daily. Clean ingestion (tickers, dates, fields), missing-value handling, and consistent outputs are step zero of every pipeline.

**Other scenarios**

* Swap to FRED (macro), MT5/IB (execution), or crypto APIs — same pattern.
* Batch backtests across many tickers with the same loader.

```python
# --- Choose assets and time range (feel free to customize) ---
TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA", "SPY"]  # 4 stocks + market ETF
START, END = "2018-01-01", "2025-01-01"

# --- Download OHLCV; we'll use Close for daily studies ---
raw = yf.download(TICKERS, start=START, end=END)

# --- Keep only Close columns, drop full-NaN rows (days markets closed) ---
close = raw["Close"].dropna(how="all")

# --- Sanity check ---
display(close.tail(3))
print("Shape:", close.shape)

# --- Persist a CSV for reproducibility and paper appendix ---
close.to_csv(f"docs/close_{START}_{END}.csv")
```

> **Note:** We use **Close** instead of Adjusted Close because Adjusted Close isn’t always available uniformly; for **directional** daily returns and cross-asset comparison, Close is acceptable.

---

## 2) Returns vs Log-Returns

**Concept**

* **Simple return** $r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ — intuitive % change.
* **Log return** $\ell_t = \ln(P_t/P_{t-1})$ — **additive over time**, plays nicely in math/stat models.

**Why quants care**

* Most models (portfolio math, factor models, time-series) like **log-returns**.
* For reporting and intuition, **simple returns** are often shown.
* Always know which one you’re using and why.

**Other scenarios**

* Multi-period compounding: sum of log-returns ≈ log(total return).
* Vol targeting and forecasts often use log-returns internally.

```python
# --- Simple returns (pct change) and log-returns ---
ret = close.pct_change().dropna()                     # simple returns
logret = np.log(close / close.shift(1)).dropna()      # log-returns

# --- Quick glance: last few rows ---
display(ret.tail(3))
display(logret.tail(3))

# --- Visualize returns for one or two assets ---
ret[["AAPL","SPY"]].plot(title="Daily Simple Returns: AAPL vs SPY")
save_fig("returns_aapl_spy"); plt.show()
```

**Key checks to internalize**

* Mean of daily returns is tiny; **volatility dominates** daily behavior.
* Log-returns and simple returns are very close at daily horizons; they diverge over long horizons or huge moves.

---

## 3) Annualization & Rolling Statistics

**Concept**

* **Rolling mean/std**: compute stats over a moving window (e.g., last 30 days) to track **time-varying risk**.
* **Annualization** for daily data: multiply mean by 252 and std by $\sqrt{252}$ (≈ trading days/year).

**Why quants care**

* Volatility **clusters**: risk regimes (calm vs stormy) are the rule, not the exception.
* Rolling stats power everything from **risk parity** to **vol-targeting**.

**Other scenarios**

* Rolling correlations (see next), rolling betas, rolling drawdowns.

```python
WINDOW = 30  # 30 trading days ≈ 1.5 months

# --- Rolling mean/std on simple returns ---
roll_mean = ret.rolling(WINDOW).mean() * 252          # annualized mean
roll_vol  = ret.rolling(WINDOW).std()  * np.sqrt(252) # annualized vol

# --- Plot rolling volatility to see clustering (choose a couple assets) ---
roll_vol[["AAPL","SPY"]].plot(title=f"{WINDOW}-Day Rolling Volatility (Annualized)")
save_fig("rolling_vol_aapl_spy"); plt.show()

# --- Optional: export rolling stats for later analysis ---
roll_vol.to_csv(f"docs/rolling_vol_{WINDOW}d_{START}_{END}.csv")
```

> **Interpretation:** Notice long stretches of high vol (e.g., crises) and calm periods (e.g., stable markets). This non-constant variance is exactly why **GARCH** exists (Project 5).

---

## 4) Correlation & Diversification

**Concept**
**Correlation** measures co-movement (−1 to +1). Lower correlations between assets = better **diversification**.

**Why quants care**

* Portfolios are built on **covariance matrices** (correlations × volatilities).
* Correlations **change over time**, especially in stress (they often rise → diversification breaks).

**Other scenarios**

* Factor modeling (market, growth, value, momentum) → correlations drive exposures.
* Hedging decisions (pair a high-corr hedge vs low-corr diversifier).

```python
# --- Static correlation across the full window ---
corr = ret.corr()

# --- Heatmap visualization ---
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Daily Returns)")
save_fig("corr_heatmap"); plt.show()

# --- Rolling correlation between two assets (e.g., AAPL vs SPY) ---
roll_corr = ret["AAPL"].rolling(WINDOW).corr(ret["SPY"])
roll_corr.plot(title=f"{WINDOW}-Day Rolling Correlation: AAPL vs SPY", ylim=(-1,1))
save_fig("rolling_corr_aapl_spy"); plt.show()
```

> **Interpretation:** Static correlation is a summary; **rolling correlation** reveals **regime shifts**. Adjust positions when correlations move (e.g., increase hedges when stock/stock correlations spike).

---

## 5) Distributions: Histograms, Fat Tails, Skewness, Kurtosis

**Concepts**

* **Fat tails:** more extreme moves than a normal (Gaussian) would predict.
* **Skewness:** asymmetry (negative skew → crashes are more severe than rallies).
* **Excess kurtosis:** heavy tails + peaky center (many small moves, some huge shocks).
* **Normality tests (Jarque–Bera):** check if distribution deviates from normal.

**Why quants care**

* **Risk management:** VaR/ES under normal assumptions *underestimates* crash risk.
* **Option pricing, stress testing, tail hedging:** fat tails dominate P\&L.
* **Strategy design:** negative skew strategies (short vol) have attractive average returns with **ugly tail risk**.

**Other scenarios**

* Crypto and small caps: heavy tails and skew are common.
* Earnings or macro events: spike tails even more.

```python
asset = "AAPL"  # analyze one asset; iterate if you want
x = ret[asset].dropna()

# --- Summary stats ---
mu, sigma = x.mean()*252, x.std()*np.sqrt(252)  # annualized
sk, exk = skew(x), kurtosis(x, fisher=True)     # fisher=True => excess kurtosis
jb_stat, jb_p = jarque_bera(x)

print(f"{asset} annualized mean={mu:.3%}, vol={sigma:.3%}")
print(f"Skew={sk:.3f}, Excess Kurtosis={exk:.3f}, Jarque-Bera p={jb_p:.4g}")

# --- Histogram of returns ---
plt.hist(x, bins=60, density=True, alpha=0.8)
plt.title(f"{asset} Daily Returns: Histogram")
save_fig(f"hist_{asset}"); plt.show()

# (Optional) Overlay a normal pdf for comparison
import numpy as np
from scipy.stats import norm
grid = np.linspace(x.min()*1.2, x.max()*1.2, 500)
plt.hist(x, bins=60, density=True, alpha=0.4, label="Empirical")
plt.plot(grid, norm.pdf(grid, loc=x.mean(), scale=x.std()), lw=2, label="Normal fit")
plt.title(f"{asset} Returns vs Normal")
plt.legend(); save_fig(f"hist_vs_normal_{asset}"); plt.show()
```

> **Interpretation:** You’ll almost always see **fatter tails** and often **negative skew** in equities. This alone explains why purely Gaussian models routinely **underprice risk**.

---

## 6) Drawdowns (Bonus but essential)

**Concept**
**Drawdown** measures peak-to-trough decline in equity; it’s how pain is felt.

**Why quants care**

* Investors (and your mental capital) experience **drawdowns**, not variance.
* Two strategies with equal Sharpe can have very different **max drawdown** profiles.

**Other scenarios**

* Compare strategies by **Ulcer Index**, **Calmar ratio** (CAGR / maxDD).
* Risk limits and kill-switches are set on drawdown behavior.

```python
# --- Simple equity curve for one asset (no dividends/fees) ---
equity = (1 + ret["SPY"]).cumprod()

# --- Compute running max and drawdown ---
running_max = equity.cummax()
drawdown = equity / running_max - 1.0

fig, ax = plt.subplots(2, 1, figsize=(12,7), sharex=True)
equity.plot(ax=ax[0], title="SPY Cumulative Return (No Fees/Divs)")
drawdown.plot(ax=ax[1], color="tomato", title="SPY Drawdown")
plt.tight_layout(); save_fig("spy_equity_drawdown"); plt.show()

print("Max drawdown:", f"{drawdown.min():.2%}")
```

---

## 7) Export: Keep Your Work Reproducible

```python
# --- Save core outputs for your paper/review ---
ret.to_csv(f"docs/returns_{START}_{END}.csv")
logret.to_csv(f"docs/logreturns_{START}_{END}.csv")
corr.to_csv(f"docs/corr_{START}_{END}.csv")
```

---

## 8) Why Quants Care — Summary (with Real Use-Cases)

1. **Returns vs Log-Returns**

* *Why:* Every model uses returns; log-returns are additive and model-friendly.
* *Use-Cases:* Portfolio math, factor modeling, time-series forecasting, option-implied return aggregation.

2. **Rolling Volatility (Risk Regimes)**

* *Why:* Volatility **clusters**; risk isn’t constant.
* *Use-Cases:* **Vol-targeting** (scale positions by recent vol), regime-aware stop losses, dynamic leverage.

3. **Correlation & Rolling Correlation**

* *Why:* Diversification depends on correlation, not asset count.
* *Use-Cases:* Asset allocation (Markowitz, risk parity), hedging, factor exposure control.
* *Note:* Correlations tend to **rise in crises** → diversification fails when you need it most.

4. **Distribution Shape (Fat Tails, Skew, Kurtosis)**

* *Why:* Tails and asymmetry drive **blow-ups** and **tail hedging** needs.
* *Use-Cases:* Tail-risk hedging, stress testing, realistic VaR/ES, strategy choice (e.g., avoid short-vol unless hedged).

5. **Drawdowns**

* *Why:* Investors feel drawdowns, not variance.
* *Use-Cases:* Risk limits (maxDD), **kill switches**, position throttling, capital allocation across strategies.

---

## 9) Common Pitfalls & Best Practices

* **Adjusted vs Close:** For total-return studies, use Adjusted Close; for daily direction and cross-sectional simplicity, **Close** is fine (consistent availability).
* **Missing data:** Align on trading days; don’t accidentally compare assets on different calendars.
* **Annualization:** Use **252** trading days; be consistent.
* **Outliers:** Don’t clip without logging; outliers are often the **signal** (fat tails).
* **Stationarity:** Prices are non-stationary; returns are closer to stationary → analyze returns.
* **Lookahead bias:** All rolling stats must use **past-only** windows (which we did).

---

## 10) Mini “Algorithms” Explained

* **`pct_change()`**: computes $(P_t - P_{t-1}) / P_{t-1}$, i.e., simple returns.
* **`np.log(close/close.shift(1))`**: log-returns (time-additive).
* **`.rolling(WINDOW).std()`**: sliding window standard deviation (recent risk).
* **Annualization**: multiply mean by 252, std by $\sqrt{252}$.
* **`.corr()`**: Pearson correlation matrix across columns (assets).
* **`skew()/kurtosis()`**: third and fourth standardized moments (shape of distribution).
* **Drawdown logic**: equity curve → running max → ratio minus 1 = drawdown.

---

## 11) TL;DR (what to memorize)

* **Analyze returns, not prices.**
* **Log-returns** add; use them for modeling.
* **Volatility clusters** → risk isn’t constant (Project 5: GARCH).
* **Correlations move** → diversification is dynamic.
* **Distributions are fat-tailed & skewed** → normal models underprice risk.
* **Drawdowns matter most** → manage to pain, not just variance.

---

### Optional: Quick “All-in-One” chart cell (handy for your paper)

```python
# 1) Rolling vol
roll_vol[["AAPL","SPY"]].plot(title=f"{WINDOW}-Day Rolling Volatility (Annualized)")
save_fig("summary_rolling_vol"); plt.show()

# 2) Correlation heatmap
sns.heatmap(ret.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Daily Returns)")
save_fig("summary_corr_heatmap"); plt.show()

# 3) Distribution for SPY
x = ret["SPY"].dropna()
plt.hist(x, bins=60, density=True, alpha=0.5, label="Empirical")
grid = np.linspace(x.min()*1.5, x.max()*1.5, 500)
from scipy.stats import norm
plt.plot(grid, norm.pdf(grid, loc=x.mean(), scale=x.std()), lw=2, label="Normal fit")
plt.title("SPY Daily Returns vs Normal")
plt.legend(); save_fig("summary_spy_hist"); plt.show()
```

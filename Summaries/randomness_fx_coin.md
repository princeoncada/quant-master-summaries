# Randomness in FX vs Coin Flips

**Why it matters to quants + commented code for every test**

**Core question:** Is EUR/USD’s **direction** (up/down each day) like a **fair coin**, or is there **structure** (bias, memory, regimes)?
**Big picture:** If direction is random, **edge from sign-prediction is hard**. If volatility/regimes have structure, **risk models & regime models can add value**.

---

## 0) Setup & Data → Returns → Sign Sequence

**Why quants care**

* All modeling starts from **returns**, not prices.
* **Sign sequences** (1=up, 0=down) let us test “random walk” cleanly.
* Separating **direction** (often near-random) from **volatility** (often structured) is essential for **risk control** and **position sizing**.

**Other scenarios**

* Crypto daily/weekly directions (BTC, ETH).
* Equity index drift vs single-stock noise.
* Futures roll strategies where direction ≈ noise but **volatility timing** matters.

```python
!pip -q install yfinance statsmodels scipy

import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
import yfinance as yf
from scipy.stats import binomtest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

plt.rcParams["figure.figsize"] = (12,5)
plt.rcParams["axes.grid"] = True
os.makedirs("docs/charts", exist_ok=True)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"docs/charts/{name}.png", dpi=300, bbox_inches="tight")

# --- Download EURUSD close prices
px = yf.download("EURUSD=X", start="2010-01-01", end="2025-01-01")["Close"].dropna()

# --- Convert to daily % returns; remove NaNs from the first diff
ret = px.pct_change().dropna()

# --- Remove exact-zero returns so sign is strictly {0,1}
ret = ret[ret != 0.0]

# --- Directional labels: 1=up day, 0=down day
sign = np.where(ret > 0, 1, 0).astype(int)

print("Obs:", len(sign), "Up-rate:", round(sign.mean(),4))
ret.plot(title="EUR/USD Daily Returns"); save_fig("eurusd_returns"); plt.show()
```

---

## 1) Bias Test (Binomial)

**Purpose here**
Is the share of **up days = 50%**? If yes, direction shows no drift.

**Why quants care (with examples)**

* **Equity indices** often have >50% up days (long-run risk premium) → explains **why buy-and-hold works**.
* **FX pairs** often \~50/50 → **carry/vol/flow** matter more than sign prediction.
* **Commodities** can show asymmetric behavior around inventory cycles.

**Other scenarios**

* Check if **intraday bar** close-to-close is biased (e.g., open-close vs close-open).
* Regime checking (bull markets might show short-term >50% up-day ratio).

```python
# k = # of ups, n = total days; test H0: p=0.5
n, k = len(sign), int(sign.sum())
res = binomtest(k, n, p=0.5)  # exact binomial test, two-sided
print(f"Up days {k}/{n} = {k/n:.4f}, p={res.pvalue:.4g}")
# → If p>0.05: cannot reject fairness → no directional bias
```

---

## 2) Runs Test (Wald–Wolfowitz)

**Purpose here**
Is the **ordering** of ups/downs random? Too many/few runs imply **over-alternation** or **streak clustering**.

**Why quants care (with examples)**

* **Momentum/mean-reversion** signals show up in **run structure**.
* **Regime detection**: persistent bull/bear regimes reduce number of runs.

**Other scenarios**

* High-frequency data to detect **quote-stuffing**/ping-pong alternation.
* Post-event windows (earnings, macro releases): do signs cluster?

```python
def count_runs(seq):
    # Count maximal blocks of identical outcomes (e.g., 11100 -> 2 runs)
    runs = 1
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            runs += 1
    return runs

def runs_test(seq):
    # Wald–Wolfowitz: compare observed runs to expected under randomness
    seq = np.asarray(seq)
    n1, n2 = int(seq.sum()), int(len(seq)-seq.sum())  # ups, downs
    R = count_runs(seq)
    # Expected runs under H0:
    mu = 1 + (2*n1*n2)/(n1+n2)
    # Variance under H0:
    var = (2*n1*n2*(2*n1*n2 - n1 - n2))/(((n1+n2)**2)*(n1+n2-1))
    # z-score and p-value
    from scipy.stats import norm
    z = (R - mu)/np.sqrt(var) if var>0 else np.nan
    p = 2*(1 - norm.cdf(abs(z)))
    return {"R":R, "E_mu":mu, "Var":var, "z":z, "pvalue":p, "n1":n1, "n2":n2}

runs_res = runs_test(sign)
runs_res
# → p>0.05: overall streak frequency consistent with randomness
```

---

## 3) Longest Streak (Monte Carlo)

**Purpose here**
Is the **maximum run** (e.g., 11 consecutive ups) unusually long for a random coin of the same length?

**Why quants care (with examples)**

* **Trend followers**: extreme persistence supports breakout rules; lack of it warns about **false optimism**.
* **Risk**: knowing what “extreme streaks” look like under H0 helps assess if a backtest’s **hot streak** is just luck.

**Other scenarios**

* Validate **Sharpe spikes** in a backtest by checking if streaks are **statistically ordinary**.
* Stress-test **risk of ruin** by simulating losing streaks.

```python
def run_lengths(seq):
    # Produce lengths of each consecutive block (e.g., 11100 -> [3,2])
    lens, cur = [], 1
    for i in range(1, len(seq)):
        if seq[i]==seq[i-1]:
            cur += 1
        else:
            lens.append(cur); cur=1
    lens.append(cur)
    return lens

obs_longest = max(run_lengths(sign))  # market's longest streak

def sim_longest_run(n, trials=3000, seed=42):
    # Monte Carlo: simulate fair coins, record each trial's longest run
    rng = np.random.default_rng(seed)
    L = np.empty(trials, dtype=int)
    for t in range(trials):
        s = rng.integers(0,2,n)      # 0/1 fair coin
        L[t] = max(run_lengths(s))   # longest run in this sim
    return L

simL = sim_longest_run(len(sign), 3000)
p_mc = (simL >= obs_longest).mean()   # MC p-value: how often coin ≥ observed
print("FX longest:", obs_longest, "MC p≈", round(p_mc,4))

# Plot to visualize where the FX longest streak sits
plt.hist(simL, bins=30, alpha=0.75, label="Coin sims")
plt.axvline(obs_longest, color="r", ls="--", label=f"FX longest={obs_longest}")
plt.title("Longest Streak: FX vs Coin (Monte Carlo)")
plt.legend(); save_fig("longest_streaks"); plt.show()
```

---

## 4) Markov Chain Memory (1-step)

**Purpose here**
Does **yesterday’s** direction change **today’s odds**? In a coin:
$P(\text{up|up}) = P(\text{up|down}) = 0.5$.

**Why quants care (with examples)**

* **Regime switching** (Markov) models: state-dependent probabilities (trend vs mean-revert regimes).
* **Execution**: short-term order-flow can create state dependence in microstructure data.

**Other scenarios**

* Credit transition matrices (AAA→AA, etc.) are Markov; emissions modeling; macro regime switches.

```python
# Count transitions between consecutive days
uu=ud=du=dd=0
for a,b in zip(sign[:-1], sign[1:]):
    if a==1 and b==1: uu+=1  # up→up
    if a==1 and b==0: ud+=1  # up→down
    if a==0 and b==1: du+=1  # down→up
    if a==0 and b==0: dd+=1  # down→down

P_up_after_up   = uu/(uu+ud)  # conditional probability
P_up_after_down = du/(du+dd)

# Test each conditional probability against 0.5
p1 = binomtest(uu, uu+ud, 0.5).pvalue
p2 = binomtest(du, du+dd, 0.5).pvalue

print("P(up|up)  =", round(P_up_after_up,4), "p=", round(p1,4))
print("P(up|down)=", round(P_up_after_down,4), "p=", round(p2,4))

plt.bar(["P(up|up)","P(up|down)"], [P_up_after_up, P_up_after_down])
plt.axhline(0.5, color="r", ls="--"); plt.ylim(0.4,0.6)
plt.title("Markov Memory Test"); save_fig("markov_memory"); plt.show()
# → If both ≈0.5 (and ns), no short-term memory in direction
```

---

## 5) Autocorrelation & Ljung–Box (Returns & Squared Returns)

**Purpose here**

* Returns ACF: is there **linear predictability** in r\_t?
* Squared returns ACF: **volatility clustering** (dependence in r\_t²).
* Ljung–Box: test multiple lags jointly ≠ 0.

**Why quants care (with examples)**

* **Before ARIMA/GARCH**, diagnose if linear/variance structure exists.
* **Residual checks**: after fitting a model, residuals should look like **white noise** (LB p-values large).
* **Volatility timing**: vol clusters → **vol-targeting** improves Sharpe / drawdowns.

**Other scenarios**

* HFT alpha decay (ACF at very short lags).
* Macro indices where volatility clusters around events (FOMC, CPI).

```python
lags = 20

# --- ACF of returns (linear dependence)
acf_r  = acf(ret, nlags=lags, fft=True)

# --- ACF of squared returns (variance dependence)
acf_r2 = acf(ret**2, nlags=lags, fft=True)

plt.stem(range(lags+1), acf_r, use_line_collection=True)
plt.title("ACF of Returns"); save_fig("acf_returns"); plt.show()

plt.stem(range(lags+1), acf_r2, use_line_collection=True)
plt.title("ACF of Squared Returns"); save_fig("acf_returns_squared"); plt.show()

# --- Ljung–Box on returns: expect large p if white noise
lb_r  = acorr_ljungbox(ret,    lags=[10,20], return_df=True)
# --- Ljung–Box on squared returns: small p → vol clustering
lb_r2 = acorr_ljungbox(ret**2, lags=[10,20], return_df=True)

print("Ljung–Box (returns):\n", lb_r, "\n")
print("Ljung–Box (squared returns):\n", lb_r2)
```

---

## 6) Hurst Exponent (R/S method)

**Purpose here**
Test **long-term memory** (persistence vs mean-reversion). H≈0.5 random, H>0.5 persistent.

**Why quants care (with examples)**

* **Strategy class selection**: If a market is persistent at your horizon → trend; if <0.5 → mean-revert.
* **Fractal markets**: scaling of volatility with horizon.

**Other scenarios**

* **Volatility** Hurst can differ from **returns** Hurst.
* Cross-asset comparison: commodities sometimes show stronger persistence.

```python
def hurst_rs(x, min_win=10, max_win=300):
    # R/S method: for each window w, average (range of cumulative demeaned series) / stdev
    x = np.asarray(x); N = len(x)
    wins = np.unique(np.logspace(np.log10(min_win), np.log10(max_win), 12).astype(int))
    RS, n = [], []
    for w in wins:
        if w*5 > N: break        # need multiple segments per window size
        m = N // w
        segs = x[:m*w].reshape(m, w)
        ratios = []
        for seg in segs:
            Y = seg - seg.mean() # de-mean
            Z = np.cumsum(Y)     # cumulative sum (profile)
            R = Z.max() - Z.min()
            S = seg.std(ddof=1)
            if S > 0: ratios.append(R/S)
        if ratios:
            RS.append(np.mean(ratios)); n.append(w)
    n, RS = np.array(n), np.array(RS)
    # slope of log(R/S) ~ H * log(n)
    H = np.polyfit(np.log(n), np.log(RS), 1)[0]
    return H, n, RS

H, n, RS = hurst_rs(ret.values)
plt.plot(np.log(n), np.log(RS), "o-")
plt.title(f"Hurst log-log fit, H ≈ {H:.3f}")
save_fig("hurst_fit"); plt.show()
print("Hurst exponent ≈", round(H,3))
```

---

## 7) Entropy (Shannon)

**Purpose here**
Measure **unpredictability** of the up/down process. Max 1 bit = perfect randomness.

**Why quants care (with examples)**

* **Feature selection** in ML: does a feature reduce entropy (add information)?
* **Market efficiency**: high entropy in direction → harder to forecast; seek **non-directional** edges (carry, term structure, cross-section).

**Other scenarios**

* Compare entropy across **timeframes**: daily may be \~1.0, intraday not.
* Compare across **assets**: some small-caps might have lower entropy.

```python
p_up = sign.mean()  # empirical probability of an up day
# Shannon entropy (base 2): H = -sum p*log2(p)
H_bits = - (p_up*np.log2(p_up) + (1-p_up)*np.log2(1-p_up))
print(f"Shannon entropy ≈ {H_bits:.3f} bits (max=1.0)")
```

---

## 8) Permutation Test (Toy 1-step Momentum)

**Purpose here**
Is a naive classifier (“today = yesterday”) **better than chance**?
Permutation test guards against **spurious backtest edges**.

**Why quants care (with examples)**

* **Backtest validation**: shuffle labels to test if “edge” survives.
* **Data snooping** defense: a real edge should beat **random relabeling**.

**Other scenarios**

* Any rule (e.g., “trade sign = 5-day sign”) can be permutation-tested.
* Use on **cross-sectional** signals (shuffle ranks).

```python
# --- Strategy prediction: today's sign = yesterday's sign
pred = np.r_[np.nan, sign[:-1]]         # shift by 1 day
acc_fx = np.mean(pred[1:] == sign[1:])  # accuracy vs true
print("FX accuracy:", round(acc_fx,3))

def perm_test_accuracy(seq, trials=3000, seed=42):
    # Shuffle labels to destroy structure; recompute accuracy each time
    rng = np.random.default_rng(seed)
    arr = np.asarray(seq)
    acc = np.empty(trials)
    for t in range(trials):
        s = rng.permutation(arr)
        p = np.r_[np.nan, s[:-1]]
        acc[t] = np.mean(p[1:] == s[1:])
    return acc

acc_sim = perm_test_accuracy(sign, 3000)
p_perm = (acc_sim >= acc_fx).mean()  # how often shuffled ≥ real accuracy
print("Permutation p≈", round(p_perm,4))

plt.hist(acc_sim, bins=40, alpha=0.75)
plt.axvline(acc_fx, color="r", ls="--", label=f"FX acc={acc_fx:.3f}")
plt.legend(); plt.title("Permutation: 1-step momentum accuracy")
save_fig("perm_test_momentum"); plt.show()
```

---

## 9) What these results *mean* for a quant

1. **Direction ≈ Random**

* No bias, no state-dependence, entropy ≈ 1 → **avoid sign-prediction toys**.
* Example: a simple “long if yesterday up” signal is **statistically void**.

2. **Volatility is Structured**

* Strong ACF in r² → **vol-targeting, GARCH**, regime models help.
* Example: **volatility targeting** (scale position to target σ) often improves Sharpe & maxDD.

3. **Persistence is Mild**

* H ≈ 0.55 → weak trending at longer horizons.
* Example: weekly/monthly trend filters may beat daily sign rules.

4. **How to act on this**

* Build **edges outside direction**: carry (rates/roll), cross-section, term structure, seasonality.
* Model **risk** carefully: vol regimes, drawdown controls, Kelly-aware sizing.

---

## 10) Common pitfalls & best practices

* **Multiple testing**: don’t hunt for the 1 significant p among 20 tests. Read the **pattern** of evidence.
* **Sample dependence**: repeat on different windows (pre/post 2020).
* **Stationarity**: sign processes can be stationary while levels are not.
* **Zeros in returns**: drop them before sign coding.
* **ACF/LB on residuals**: after ARIMA/GARCH, **re-test residuals** to confirm whiteness.

---

## 11) TL;DR for memory

* **Direction**: coin-like → no simple edge.
* **Volatility**: clusters → model it (GARCH, vol-targeting).
* **Long memory**: slight persistence → trend at longer horizons.
* **Use permutation & Monte Carlo**: to keep yourself honest.

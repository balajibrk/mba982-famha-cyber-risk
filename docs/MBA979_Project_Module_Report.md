# MBA979 — Project Module Report

**Empirical Asset Pricing via Machine Learning:  
A Reproduction and Adaptation on Indian Equities (Nifty 50)**

| Field | Detail |
|-------|--------|
| **Course** | MBA979 — Project Module |
| **Report type** | Project Review / Module Report |
| **Base paper** | Gu, Kelly, and Xiu (2020), *Empirical Asset Pricing via Machine Learning*, *The Review of Financial Studies*, 33(5), 2223–2273 |
| **Market studied** | India — Nifty 50 equities |
| **Sample period** | January 2015 – December 2024 |
| **Code repository** | https://github.com/balajibrk/gkx-india |
| **Implementation** | Python (scikit-learn, LightGBM, PyTorch) |

---

## 1. Executive Summary

This project reproduces the core empirical framework of Gu, Kelly, and Xiu (2020) — hereafter GKX (2020) — and adapts it to the Indian equity market. Using monthly data on Nifty 50 constituents over 2015–2024, we construct a panel of stock characteristics interacted with India-relevant macroeconomic variables, train nine return-prediction models spanning linear, tree-based, and neural-network families, and evaluate them with out-of-sample \(R^2\), information coefficients, Diebold–Mariano tests, and long–short portfolio performance.

**Main finding.** Consistent with GKX (2020), the deepest neural network (NN3) achieves the highest out-of-sample predictive \(R^2\) on the 2022–2024 test sample (**+1.0458%**), outperforming the OLS-3 baseline (**+0.6458%**). Macro–characteristic interaction features dominate variable importance. Portfolio results are noisier on a 47-stock universe: PCR yields the strongest long–short Sharpe ratio (**0.79**), while NN3’s ranking signal is weaker despite its superior \(R^2\) — a known small-sample tension between squared-error fit and cross-sectional ranking quality.

All code, metric tables, prediction files, and figures are available in the public repository linked above.

---

## 2. Base Paper: Gu, Kelly, and Xiu (2020)

### 2.1 Bibliographic details

> Gu, Shihao, Bryan Kelly, and Dacheng Xiu. 2020. “Empirical Asset Pricing via Machine Learning.” *The Review of Financial Studies* 33 (5): 2223–2273. https://doi.org/10.1093/rfs/hhaa009

### 2.2 Abstract of the base paper (summary)

GKX (2020) study the problem of predicting the cross-section of individual U.S. stock returns using a large set of firm characteristics and their interactions with macroeconomic state variables. They compare a wide range of machine-learning methods — including ordinary least squares with few predictors, penalised linear models (elastic net), dimension-reduction methods (principal component regression and partial least squares), regression trees and boosted trees, and multilayer neural networks — against a common empirical design.

The paper’s central conclusions are:

1. **Predictive gains from ML.** Flexible nonlinear methods, especially neural networks and gradient-boosted trees, produce substantially higher out-of-sample \(R^2\) than simple linear benchmarks.
2. **Economic value.** Predictions translate into economically meaningful long–short portfolio returns and Sharpe ratios.
3. **Which signals matter.** A relatively small subset of predictors — especially momentum-related characteristics and interactions with macro variables — accounts for most of the predictive content.
4. **Regularisation is essential.** Unconstrained high-dimensional OLS fails; shrinkage, dimension reduction, and ensemble methods are necessary to control overfitting.

The original study uses a long U.S. panel (roughly 1957–2016), a very large cross-section of stocks, and dozens of CRSP/Compustat-style characteristics. Our project keeps the *methodological skeleton* of GKX and transplants it to a shorter, smaller Indian sample with India-specific macros.

### 2.3 What this project reproduces vs. adapts

| Dimension | GKX (2020) — U.S. | This project — India |
|-----------|-------------------|----------------------|
| Universe | Broad CRSP equities (~thousands) | Nifty 50 (≈47 stocks with usable history) |
| Period | Multi-decade (≈1957–2016) | **Jan 2015 – Dec 2024** |
| Characteristics | ≈94 firm signals | **17** price/volume-based characteristics |
| Macro variables | U.S. macro states | **8** India-focused macros |
| Feature design | Char × macro interactions | Same logic → **153** features |
| Models | OLS-3, ENet, PCR, PLS, RF, GBRT, NN1–NN5 (paper) | OLS-3, ENet, PCR, PLS, RF, GBRT, **NN1–NN3** |
| Portfolio sorts | Deciles | **Quintiles** (appropriate for ~47 names) |
| Primary metric | \(R^2_{\text{oos}}\) vs. zero forecast | Same |
| Evaluation extras | IC, DM tests, portfolios | Same suite |

---

## 3. Research Question and Objectives

### 3.1 Research question

> Can machine-learning methods improve monthly cross-sectional return prediction for Indian large-cap equities relative to simple linear benchmarks, and does the GKX finding that nonlinear models dominate linear ones hold on a Nifty 50 sample?

### 3.2 Project objectives

1. Reconstruct the GKX empirical pipeline (features → train/val/test → model zoo → evaluation → portfolios).
2. Assemble a reproducible Indian dataset with clearly documented sources and sample periods.
3. Train and compare nine models under a common protocol.
4. Report statistical and economic performance honestly, including limitations induced by the small cross-section and short history.
5. Deliver a public, reproducible codebase for academic review.

---

## 4. Data

### 4.1 Overall sample design

| Item | Specification |
|------|---------------|
| Equity universe | Nifty 50 constituents (NSE), as listed in project config |
| Stocks successfully used | ≈47 (after data-availability filters) |
| Frequency | **Monthly** |
| Calendar coverage | **2015-01-01 to 2024-12-31** |
| Train sample | **2015-01 → 2019-12** |
| Validation sample | **2020-01 → 2021-12** |
| Out-of-sample test | **2022-01 → 2024-12** |
| Prediction target | One-month-ahead excess return |

The train / validation / test split follows the GKX logic of chronological separation: hyperparameters are chosen on validation; all headline results use the held-out test period only.

### 4.2 Equity price and return data

| Field | Detail |
|-------|--------|
| **Primary source** | Upstox API v2 (NSE equity monthly OHLCV) |
| **Fallback source** | Yahoo Finance (`yfinance`) adjusted closes, if Upstox is unavailable |
| **Period requested** | 2015-01-01 – 2024-12-31 |
| **Instruments** | Nifty 50 symbols (e.g. RELIANCE, TCS, HDFCBANK, INFY, …) |
| **Processed object** | Monthly panel of prices → simple returns → excess returns |

Excess returns are formed relative to a market/risk-free proxy constructed in the feature pipeline (market return from the cross-section / index side of the build). All characteristics are **lagged by one month** to avoid look-ahead bias.

### 4.3 Macroeconomic and market-state variables

Eight macro (or market-state) series are aligned to the same monthly grid **2015-01 to 2024-12**:

| # | Variable | Description | Source | Identifier / ticker | Period |
|---|----------|-------------|--------|---------------------|--------|
| 1 | `repo_rate` | India long-term interest-rate proxy (scaled) | **FRED** (via `pandas-datareader`) | `INDIRLTLT01STM` | 2015–2024 |
| 2 | `india_vix` | India VIX (volatility / fear index) | **Yahoo Finance** | `^INDIAVIX` | 2015–2024 |
| 3 | `inr_usd` | INR per USD exchange rate | **Yahoo Finance** | `INR=X` | 2015–2024 |
| 4 | `cpi_yoy` | India CPI year-over-year inflation | **FRED** | `INDCPIALLMINMEI` (12-month pct change) | 2015–2024 |
| 5 | `nifty_pe` | Nifty aggregate P/E | Manual NSE CSV if present; else calibrated synthetic series | local / synthetic | 2015–2024 |
| 6 | `nifty_pb` | Nifty aggregate P/B | Same as above | local / synthetic | 2015–2024 |
| 7 | `ep_macro` | Aggregate earnings yield | Derived | \(1 / \texttt{nifty\_pe}\) | 2015–2024 |
| 8 | `bm_macro` | Aggregate book-to-market | Derived | \(1 / \texttt{nifty\_pb}\) | 2015–2024 |

Macro series are month-end aligned, forward-filled where needed for sparse months, and standardised before interaction with characteristics.

> **Note for review:** If an NSE PE/PB CSV is not supplied, the pipeline falls back to a synthetic PE/PB path (documented in code). Interest-rate and inflation series prefer FRED; FX and India VIX prefer Yahoo Finance. Equity prices prefer Upstox with Yahoo Finance fallback.

### 4.4 Firm characteristics (17)

Following the spirit of GKX Section 2.1, but restricted to signals computable from the price/volume panel (no full Compustat-style fundamentals):

| Group | Characteristics |
|-------|-----------------|
| Momentum / reversal | `mom1m`, `mom3m`, `mom6m`, `mom12m`, `mom36m`, `reversal` |
| Risk / volatility | `beta`, `retvol`, `idiovol`, `maxret` |
| Liquidity / trading | `turnover`, `turnover_vol`, `dolvol` (and related volume features as implemented) |
| Valuation proxies from prices | size / market-equity style signals where available from the panel |

Each characteristic is:

1. Computed with a **one-month lag**.
2. **Cross-sectionally rank-normalised** each month to the interval \([-1, 1]\).

### 4.5 Final feature matrix

\[
Z_{i,t} \;=\; \bigl[\; c_{i,t},\;\; c_{i,t}\otimes x_t \;\bigr]
\]

- 17 standalone characteristics \(c_{i,t}\)
- 17 × 8 = 136 interaction terms with macros \(x_t\)
- **Total: 153 features**

Missing feature values after construction are filled with zeros (post rank-normalisation), preserving panel balance for ML training.

---

## 5. Methodology (Step by Step)

This section walks through the full empirical pipeline exactly as implemented in the repository.

### Step 1 — Fetch equity prices (`data/01_fetch_prices.py`)

1. Load the Nifty 50 symbol list from `config/settings.py`.
2. For each symbol, request monthly OHLCV from Upstox API v2 for 2015–2024.
3. Resolve instrument keys via ISIN search when symbol-only keys fail.
4. If Upstox fails for a name, fall back to Yahoo Finance.
5. Cache per-symbol files under `data/raw/` (local only; not published) and build a combined returns panel.

### Step 2 — Fetch macro panel (`data/02_fetch_macro.py`)

1. Download FRED series for India rates and CPI.
2. Download Yahoo Finance series for India VIX and INR/USD.
3. Load or synthesise Nifty PE/PB; construct `ep_macro` and `bm_macro`.
4. Align all series to a common monthly index covering 2015–2024.
5. Save `macro_panel.parquet`.

### Step 3 — Build features and labels (`data/03_build_features.py`)

1. Compute 17 lagged characteristics from the returns panel.
2. Rank-normalise characteristics cross-sectionally to \([-1,1]\).
3. Standardise macros and form char × macro interactions.
4. Define the prediction label \(y_{i,t+1}\) = next-month excess return.
5. Assign chronological splits: train / val / test using the dates in Section 4.1.
6. Save `X_features.parquet`, `y_labels.parquet`, `meta.parquet`, and `feature_names.csv`.

### Step 4 — Linear models (`models/04_linear_models.py`)

Train four linear benchmarks; tune regularisation / components on the **validation** set; save test predictions.

| Model | Description |
|-------|-------------|
| **OLS-3** | Sparse OLS baseline with three economically motivated predictors (GKX-style simple benchmark) |
| **Elastic Net** | Penalised linear model (L1 + L2); \(\alpha\) and \(\ell_1\)-ratio tuned on validation |
| **PCR** | Principal component regression; number of components tuned on validation |
| **PLS** | Partial least squares; number of components tuned on validation |

### Step 5 — Tree ensembles (`models/05_tree_models.py`)

| Model | Description |
|-------|-------------|
| **Random Forest** | Bagged regression trees; depth / feature subsample hyperparameters tuned on validation |
| **GBRT (LightGBM)** | Gradient-boosted trees; learning rate, leaves, and related hyperparameters tuned on validation |

Feature importances from RF and GBRT are saved and later combined into a rank-based importance table.

### Step 6 — Neural networks (`models/06_neural_networks.py`)

Feed-forward networks following the GKX architecture family:

| Model | Hidden layers (width) |
|-------|------------------------|
| **NN1** | 32 |
| **NN2** | 32 → 16 |
| **NN3** | 32 → 16 → 8 |

Training protocol:

- Activations: ReLU  
- Batch normalisation and dropout  
- Adam optimiser with learning-rate decay  
- L1 penalty on weights  
- Gradient clipping  
- Early stopping on validation loss  
- Dropout tuned on validation (especially for NN2)  
- **10 random seeds** per architecture; ensemble average of predictions; divergent seeds filtered by validation loss  

> Training was executed on CPU in this environment (CUDA wheels were unavailable for the installed Python version). Results are numerically valid; only wall-clock time differs from a GPU run.

### Step 7 — Statistical evaluation (`eval/07_metrics.py`)

For every model on the test set we compute:

1. **Out-of-sample \(R^2\)** against a zero forecast (GKX definition):
   \[
   R^2_{\text{oos}}
   =
   1 - \frac{\sum (y - \hat y)^2}{\sum y^2}
   \]
2. **Information Coefficient (IC):** monthly Spearman rank correlation between predictions and realised returns; report mean IC, \(t\)-stat, and fraction of months with IC > 0.
3. **Diebold–Mariano (DM) test** of predictive accuracy versus OLS-3.
4. **Variable importance:** combined RF + GBRT ranks.

### Step 8 — Portfolio construction (`eval/08_portfolio.py`)

Each month in the test sample:

1. Rank stocks by model predicted return.
2. Form **quintile** portfolios (Q1 = lowest predicted, Q5 = highest).
3. Construct a long–short spread (long Q5, short Q1), value-weighted when market-cap information is available, otherwise equal-weighted.
4. Compute Sharpe ratio (annualised with \(\sqrt{12}\)), annualised mean return, cumulative return, win rate, and maximum drawdown.
5. Compare against a Nifty buy-and-hold style benchmark and estimate a simple factor-adjusted alpha where feasible.

### Step 9 — Visualisation and reporting (`viz/09_charts.py`, notebook)

Publication-style charts:

1. \(R^2_{\text{oos}}\) bar chart  
2. Variable importance (top predictors)  
3. Validation vs test \(R^2\) scatter (overfitting diagnosis)  
4. Quintile return pattern  
5. Cumulative long–short PnL  
6. India macro-interaction story  

Master narrative notebook: `notebooks/00_master_results.ipynb`.

---

## 6. Results

All figures below refer to the **out-of-sample test window: 2022–2024** unless noted. Validation metrics are used only for model selection / diagnosis.

### 6.1 Out-of-sample predictive \(R^2\)

| Model | Family | Val \(R^2\) (%) | Test \(R^2\) (%) |
|-------|--------|----------------:|-----------------:|
| OLS-3 | Linear | 1.5564 | **0.6458** |
| Elastic Net | Linear | 2.1609 | 0.4356 |
| PCR | Linear | 1.8217 | 0.4729 |
| PLS | Linear | 2.3478 | 0.0351 |
| Random Forest | Tree | 2.1393 | 0.0825 |
| GBRT | Tree | 1.0206 | −0.4166 |
| NN1 | Neural | 3.9555 | −1.4943 |
| NN2 | Neural | 4.4926 | −0.5515 |
| **NN3** | Neural | 3.6290 | **+1.0458** |

**Interpretation**

- NN3 is the best test predictor by \(R^2_{\text{oos}}\), confirming the GKX qualitative result that deeper nonlinear models can beat linear baselines.
- Several flexible models (NN1, NN2, GBRT) show **validation strength but test weakness**, illustrating overfitting risk on a short Indian sample — exactly why GKX emphasise validation-based regularisation and held-out testing.
- OLS-3 remains a surprisingly competitive linear benchmark on this small universe.

### 6.2 Information coefficients (test period)

| Model | Mean IC | \(t\)-stat | % months IC > 0 | Months |
|-------|--------:|----------:|----------------:|-------:|
| OLS-3 | 0.0092 | 0.26 | 60.0% | 35 |
| ENet | 0.0107 | 0.35 | 51.4% | 35 |
| PCR | 0.0058 | 0.15 | 48.6% | 35 |
| PLS | −0.0055 | −0.17 | 45.7% | 35 |
| RF | 0.0022 | 0.08 | 48.6% | 35 |
| GBRT | 0.0078 | 0.26 | 45.7% | 35 |
| NN1 | −0.0365 | −1.25 | 45.7% | 35 |
| NN2 | −0.0211 | −0.79 | 37.1% | 35 |
| NN3 | −0.0577 | −1.88 | 31.4% | 35 |

IC magnitudes are small and mostly insignificant — expected with only **35 test months** and ≈47 names. Notably, NN3’s mean IC is negative even though its \(R^2_{\text{oos}}\) is best. This can occur when squared-error fit improves while cross-sectional rank ordering does not; it is an important caveat for portfolio translation on small samples.

### 6.3 Diebold–Mariano tests versus OLS-3

| Model | DM statistic | \(p\)-value | Significant at 5%? |
|-------|-------------:|------------:|:------------------:|
| ENet | −0.31 | 0.75 | No |
| PCR | −0.28 | 0.78 | No |
| PLS | −0.75 | 0.45 | No |
| RF | −0.45 | 0.65 | No |
| GBRT | −0.49 | 0.62 | No |
| NN1 | −1.02 | 0.31 | No |
| NN2 | −0.45 | 0.65 | No |
| NN3 | +0.33 | 0.74 | No |

No model significantly beats OLS-3 on the DM test at conventional levels. With 35 months, the test is underpowered relative to the multi-decade GKX sample. We therefore treat \(R^2_{\text{oos}}\) ordering and economic portfolio diagnostics as complementary evidence, not as DM-rejected superiority.

### 6.4 Variable importance (India-specific pattern)

Combined RF + GBRT ranks show that **macro-interacted features dominate** the top of the importance list. Leading examples include:

1. `mom12m × india_vix`  
2. `beta × repo_rate`  
3. `retvol × ep_macro` / standalone `retvol`  
4. `maxret × inr_usd`  
5. `mom12m × ep_macro`, `mom12m × inr_usd`, `mom12m × cpi_yoy`, `mom12m × repo_rate`

**Economic reading.** Indian large-cap returns in this sample appear tightly linked to the **policy-rate / inflation / FX / fear** state of the market. Momentum’s predictive content is state-dependent — amplified when interacted with India VIX and RBI-related rate conditions — which is a natural emerging-market counterpart to GKX’s emphasis on characteristic–macro interactions.

### 6.5 Long–short quintile portfolio performance (test: 2022–2024)

| Model | Sharpe (ann.) | Ann. return | Cumulative return | Win rate | Max drawdown |
|-------|--------------:|------------:|------------------:|---------:|-------------:|
| **PCR** | **0.790** | **+11.67%** | **+36.00%** | **60.0%** | −15.46% |
| RF | 0.557 | +7.53% | +21.32% | 57.1% | −16.67% |
| GBRT | 0.123 | +1.62% | +2.18% | 51.4% | −20.33% |
| ENet | −0.198 | −2.14% | −7.64% | 48.6% | −18.28% |
| PLS | −0.227 | −3.57% | −13.11% | 48.6% | −37.00% |
| NN1 | −0.228 | −3.96% | −14.56% | 34.3% | −30.77% |
| OLS-3 | −0.286 | −4.06% | −13.78% | 48.6% | −26.11% |
| NN3 | −0.556 | −8.06% | −23.40% | 42.9% | −28.99% |
| NN2 | −0.675 | −9.31% | −25.97% | 40.0% | −35.24% |

**Interpretation**

- **PCR** produces the strongest tradable long–short record in this small-universe setting.
- **NN3** does *not* win on Sharpe despite winning on \(R^2_{\text{oos}}\), consistent with its weak/negative IC.
- For academic scoring of the GKX *prediction* claim, \(R^2_{\text{oos}}\) is the primary metric (as in the paper). Portfolio results are reported transparently as a secondary economic lens and are more fragile when \(N_{\text{stocks}}\) is small.

### 6.6 Mapping results back to GKX claims

| GKX claim | Evidence in this India project | Verdict |
|-----------|--------------------------------|---------|
| Nonlinear ML can beat linear \(R^2_{\text{oos}}\) | NN3 test \(R^2\) > OLS-3 | **Supported** |
| Regularisation / validation matter | NN1/NN2 strong in val, weak in test | **Supported** |
| Macro interactions are important | Top importances are mostly interactions | **Supported** |
| ML portfolios dominate economically | PCR strongest; NN3 weak on Sharpe/IC | **Partially supported / sample-limited** |
| DM significance vs OLS | No model significant at 5% | **Not confirmed** (short test window) |

---

## 7. Discussion

### 7.1 What transferred cleanly from the U.S. paper

The GKX pipeline — lagged characteristics, macro interactions, chronological validation, a multi-family model zoo, and \(R^2_{\text{oos}}\) versus a zero benchmark — is portable. Even with a fraction of the original data scale, the deepest network still posts the best test \(R^2\), and macro-conditioned signals dominate importance rankings.

### 7.2 What is India-specific

Unlike a U.S. all-equity CRSP universe, Nifty 50 is a **large-cap, high-visibility** index. Macro factors (policy rates, inflation, INR, India VIX) play an outsized role. The prominence of `mom12m × india_vix` and `beta × repo_rate` is consistent with a market where global risk appetite and RBI cycles mediate stock-level predictability.

### 7.3 Why \(R^2\) and portfolio ranks can disagree

\(R^2_{\text{oos}}\) rewards squared-error accuracy relative to a zero forecast. Quintile long–short performance rewards **stable cross-sectional ranking**. With ≈47 stocks and 35 months, a model can shrink overall error (good \(R^2\)) without sorting winners and losers reliably (weak IC / Sharpe). This project reports both, rather than forcing a single narrative.

---

## 8. Limitations

1. **Universe size.** ≈47 stocks vs. thousands in GKX → noisy IC and portfolio inference.  
2. **History length.** 10 years vs. multi-decade U.S. sample → DM tests underpowered.  
3. **Characteristic breadth.** 17 price/volume signals vs. ≈94 in the paper → no full fundamental set (earnings quality, accruals, analyst data, etc.).  
4. **PE/PB macro path.** May rely on synthetic fallback if NSE fundamentals CSV is absent.  
5. **Transaction costs / shorting.** Long–short spreads are theoretical; NSE shorting frictions and costs are not modelled.  
6. **Compute environment.** Neural nets trained on CPU in this run; architecture and seeds follow the paper’s spirit, but hardware differs from a CUDA production setup.

These limitations are material and should be read as scope constraints of an MBA project reproduction, not as failures of the original paper.

---

## 9. Reproducibility and Code Access

| Item | Location |
|------|----------|
| **Public GitHub repository** | https://github.com/balajibrk/gkx-india |
| Configuration / dates / tickers | `config/settings.py` |
| Data scripts | `data/01_fetch_prices.py`, `02_fetch_macro.py`, `03_build_features.py` |
| Models | `models/04_linear_models.py`, `05_tree_models.py`, `06_neural_networks.py` |
| Evaluation | `eval/07_metrics.py`, `08_portfolio.py` |
| Charts | `viz/09_charts.py`, `viz/figures/` |
| Results tables | `results/*.csv` |
| Narrative notebook | `notebooks/00_master_results.ipynb` |
| Dependencies | `requirements.txt` |

**Recommended run order**

```text
python data/01_fetch_prices.py
python data/02_fetch_macro.py
python data/03_build_features.py
python models/04_linear_models.py
python models/05_tree_models.py
python models/06_neural_networks.py
python eval/07_metrics.py
python eval/08_portfolio.py
python viz/09_charts.py
```

Secrets (API tokens) are loaded from a local `.env` file and are **not** stored in the public repository. An `.env.example` template is provided.

---

## 10. Conclusion

This MBA979 project module reproduces the empirical machine-learning asset-pricing framework of Gu, Kelly, and Xiu (2020) on Indian Nifty 50 data for **January 2015 – December 2024**. Using 153 characteristic and macro-interaction features and nine models, we find that:

1. **NN3 attains the best out-of-sample \(R^2\) (+1.0458%)**, exceeding the OLS-3 baseline — in line with the paper’s nonlinear-advantage claim.  
2. **Macro-interacted predictors dominate variable importance**, highlighting an India-specific, state-dependent return structure.  
3. **Portfolio evidence is mixed**: PCR delivers the strongest long–short Sharpe (0.79), while NN3’s ranking metrics are weak, underscoring small-sample limits when moving from statistical fit to implementable sorts.  
4. The full pipeline, results, and documentation are openly available at **https://github.com/balajibrk/gkx-india** for faculty review and verification.

Overall, the project demonstrates that the GKX methodology is implementable and informative on Indian large-cap equities, while also clarifying which of the paper’s conclusions survive — and which become fragile — under a shorter emerging-market sample.

---

## References

1. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, *33*(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009  
2. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, *13*(3), 253–263.  
3. Federal Reserve Bank of St. Louis (FRED). India interest-rate and CPI series (`INDIRLTLT01STM`, `INDCPIALLMINMEI`).  
4. National Stock Exchange of India / Yahoo Finance. Nifty-related market data and India VIX.  
5. Upstox API v2. NSE equity historical market data.

```bibtex
@article{gu2020empirical,
  title   = {Empirical Asset Pricing via Machine Learning},
  author  = {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal = {The Review of Financial Studies},
  volume  = {33},
  number  = {5},
  pages   = {2223--2273},
  year    = {2020},
  doi     = {10.1093/rfs/hhaa009}
}
```

---

## Appendix A — Repository structure (for review)

```text
gkx_india/
├── README.md
├── requirements.txt
├── config/settings.py
├── data/01_fetch_prices.py
├── data/02_fetch_macro.py
├── data/03_build_features.py
├── models/04_linear_models.py
├── models/05_tree_models.py
├── models/06_neural_networks.py
├── eval/07_metrics.py
├── eval/08_portfolio.py
├── viz/09_charts.py
├── viz/figures/                 # six result charts
├── results/                     # predictions + metric CSVs
├── notebooks/00_master_results.ipynb
└── docs/MBA979_Project_Module_Report.md   # this report
```

## Appendix B — Data source & period checklist

| Dataset | Source | Period used |
|---------|--------|-------------|
| Nifty 50 monthly OHLCV | Upstox API v2 (fallback: Yahoo Finance) | 2015-01 – 2024-12 |
| India long-term rate proxy | FRED `INDIRLTLT01STM` | 2015-01 – 2024-12 |
| India CPI | FRED `INDCPIALLMINMEI` | 2015-01 – 2024-12 |
| India VIX | Yahoo Finance `^INDIAVIX` | 2015-01 – 2024-12 |
| INR/USD | Yahoo Finance `INR=X` | 2015-01 – 2024-12 |
| Nifty PE / PB | NSE CSV or documented synthetic fallback | 2015-01 – 2024-12 |
| Train / Val / Test labels | Constructed in feature pipeline | Train 2015–2019; Val 2020–2021; Test 2022–2024 |

---

*End of MBA979 Project Module Report*

# Counterfactual Inflation Analysis: What If Ukraine Had Been Part of the Euro Area?

**Author:** Anca-Madalina Cotarta
**Course:** Quantitative Methods in Finance — Final Exam
**Programme:** M2 Finance Technology & Data, Université Paris 1 Panthéon-Sorbonne
**Academic year:** 2025–2026

---

## Overview

This repository contains the full replication package for a take-home exam
in Quantitative Methods in Finance. The paper asks a counterfactual
question:

> *What would Ukraine's inflation trajectory have looked like if it had
> been a member of the Euro Area?*

The answer is built in three layers:

1. **Part A — Monetary regime chronology (2000–2025).** A documented
   reconstruction of the National Bank of Ukraine's de facto exchange-rate
   regime, used to operationalise a time-varying *monetary sovereignty
   index* `θ_t ∈ [0,1]`.
2. **Part B — Bayoumi–Eichengreen Structural VAR.** A bivariate SVAR on
   industrial production growth and YoY inflation, identified with the
   Blanchard–Quah long-run restriction (demand shocks have no permanent
   effect on output), estimated separately for Ukraine and the Euro Area
   on the peace period 2006–2021.
3. **Counterfactual construction by *shock replacement*.** Ukraine retains
   its own idiosyncratic supply shocks but inherits the Euro Area's
   monetary anchor (contemporaneous Euro Area inflation). The result is
   then weighted by `θ_t` to ensure consistency with Part A: during de
   facto USD pegs the counterfactual stays close to actual inflation,
   during inflation-targeting and devaluation episodes it diverges.

The central finding is that Euro Area membership would have substantially
reduced Ukrainian inflation, with an average gap of **+8.3 pp** over the
sample, peaking at **+26.8 pp** during the 2014–15 currency collapse and
**+12.2 pp** during the 2016–19 inflation-targeting period, while remaining
negligible during the de facto peg years.

---

## Repository structure

```
.
├── README.md                          ← this file
├── paper.tex                          ← LaTeX source of the paper
├── paper.pdf                          ← compiled paper (16 pages)
├── ukraine_counterfactual.py          ← end-to-end replication script
├── Data.xlsx                          ← input dataset (provided)
└── output/                            ← all artefacts produced by the script
    ├── fig_main_counterfactual.png    ← main deliverable (Part B figure)
    ├── fig_uah_usd_partA.png          ← UAH/USD chronology (Part A figure)
    ├── fig_irf.png                    ← structural impulse responses + 68% CI
    ├── fig_shocks.png                 ← extracted structural shocks
    ├── partA_regime_chronology.csv    ← Part A regime table
    └── counterfactual_series.csv      ← actual vs CF inflation, monthly
```

---

## Data

All raw inputs live in `Data.xlsx`. The file contains:

| Sheet                  | Content                                                      | Source     |
| ---------------------- | ------------------------------------------------------------ | ---------- |
| `data_ukraine_cpi_raw` | Ukraine CPI as a previous-month index, 2000–2025             | SSSU       |
| `data_ecb_hicp_panel`  | YoY HICP for 11 EA countries, 2000–2025                      | ECB        |
| `monthly_data`         | Harmonised inflation, industrial production, FX, monthly     | Bloomberg  |
| `daily_data`           | Commodity and financial market series, daily                 | Bloomberg  |
| `Feuil1`               | Variable dictionary (Bloomberg tickers)                      | —          |

The estimation window starts in **May 2006** because this is the earliest
date for which monthly year-on-year industrial production growth is
available for Ukraine on Bloomberg (ticker `UAIPYY`). The Part A regime
chronology nevertheless extends back to **2000** to document the longer
institutional context, as required by the assignment.

---

## Methodology

### Part A — Monetary regime chronology

Ten regime episodes are identified between 2000 and 2025, sourced from
IMF Article IV reports, IMF AREAER classifications, and NBU policy
documents. Each episode is assigned a `Sovereignty` flag (Low / High) that
captures the intensity of the counterfactual treatment of EA membership.
This flag is converted to `θ_t ∈ {0.10, 1.00}`:

- `θ_t = 0.10` during de facto USD pegs (NBU monetary policy already
  subordinated to a foreign anchor → small treatment effect)
- `θ_t = 1.00` during devaluation episodes, IT regime, and active wartime
  policy (genuine monetary autonomy → large treatment effect)

### Part B — Bayoumi–Eichengreen SVAR

For each economy `i ∈ {UA, EA}`, a bivariate VAR is estimated on:

```
y_t = (Δq_t, π_t)'
```

where `Δq_t` is YoY industrial production growth and `π_t` is YoY
inflation. Lag length is selected by AIC: `p = 2` for Ukraine, `p = 4`
for the Euro Area.

**Identification (Blanchard–Quah).** The single long-run restriction is
that demand shocks have no permanent effect on output: the long-run
multiplier `C(1)` is lower triangular. Combined with the variance
normalisation `E[ε ε'] = I`, this pins down the structural impact matrix
`B` up to sign normalisation.

**Inference.** 68% confidence bands on the IRFs are obtained by residual
bootstrap with 500 replications.

### Counterfactual construction

The pure counterfactual inflation is

```
π_CF,t = π_EA,t  +  Σ_h ψ_h^{UA,supply} · ε_{t-h}^{s,UA}
```

i.e. Ukraine inherits the contemporaneous Euro Area inflation level
(imported ECB credibility) and keeps the dynamic effect of its own
idiosyncratic supply shocks. The `θ_t`-weighted counterfactual is

```
π̃_CF,t = (1 - θ_t) · π_actual,t  +  θ_t · π_CF,t
```

This formulation satisfies the two requirements of the assignment:
(i) it differs from a flat cross-sectional EA mean (it preserves UA-specific
supply shock dynamics); and (ii) the treatment intensity varies over time
in line with the Part A chronology.

---

## How to reproduce

### Requirements

- Python 3.12+
- `numpy`, `pandas`, `statsmodels`, `matplotlib`, `scipy`, `openpyxl`

### Run

```bash
pip install numpy pandas statsmodels matplotlib scipy openpyxl
python ukraine_counterfactual.py
```

The script runs end-to-end from `Data.xlsx` and writes all outputs (PNG
figures + CSV series) to `output/`. Expected runtime: **~30 seconds**
(the bulk being the 500-replication bootstrap for the IRF confidence
bands). The random seed is fixed at `42` for full reproducibility.

### Console output (extract)

```
[1] Loading data ...
    monthly_data shape: (228, 6), range: 2006-05-01 → 2025-04-01
[2] Stationarity (ADF) tests on full sample:
  ADF Inflation_Ukraine        : stat= -2.999, p-value= 0.035  → STATIONARY
  ADF IndProd_Ukraine (YoY)    : stat= -3.537, p-value= 0.007  → STATIONARY
[5] SVAR Ukraine ... Lag selection (AIC): p = 2
[6] SVAR Euro Area ... Lag selection (AIC): p = 4
[7] Bootstrapping IRF confidence bands (500 replications) ...
[11] Quantitative summary of the counterfactual:
    Mean gap (actual − CF, weighted): 8.30 pp
    Cumulative pre-2016 gap (peg era): 7.96 pp
    Cumulative 2016–2022 gap (IT):    9.76 pp
    War period gap (2022–2025):       6.60 pp
DONE — all outputs written to ./output/
```

---

## Headline numbers

| Sub-period                       | Actual UA | CF (weighted) | Gap     |
| -------------------------------- | --------- | ------------- | ------- |
| Pegs (2006–08, 2010–13)          | ~ 10.0    | ~ 9.6         | +0.4    |
| GFC devaluation (2008–09)        | 17.8      | 6.2           | +11.6   |
| Crimea / Donbas (2014–15)        | 30.3      | 3.5           | **+26.8** |
| Inflation Targeting (2016–19)    | 12.1      | −0.1          | +12.2   |
| COVID + late IT (2020–21)        | 6.0       | 1.1           | +4.9    |
| War-time peg (2022–23)           | 18.8      | 14.8          | +4.0    |
| Managed float (2023–25)          | 7.9       | −1.4          | +9.3    |
| **Sample average**               | **12.8**  | **4.5**       | **+8.3** |

(Averages, percentage points. See Table 4 of the paper.)

---

## Limitations

- The Blanchard–Quah identification assumes a stable VAR data-generating
  process, which is incompatible with the abrupt regime change of
  February 2022 — hence the choice to estimate on the peace period
  2006–2021 and project structural shocks onto the full sample.
- The shock-replacement scheme abstracts from financial-stability
  spillovers (De Grauwe, 2012): the counterfactual measures inflation,
  not aggregate welfare. Under EA membership, Ukraine would have avoided
  a currency crisis in 2014 and 2022 but might have faced sovereign
  debt-market stress instead.
- The bivariate VAR (output, inflation) is parsimonious. Future
  extensions could augment the system with policy rates, exchange rates
  and commodity prices, or complement the SVAR with synthetic-control
  or common-factor approaches (Abadie et al., 2010; Ciccarelli & Mojon,
  2010).

---

## Key references

Bayoumi, T. and Eichengreen, B. (1993). *Shocking aspects of European
monetary integration*.

Blanchard, O. J. and Quah, D. (1989). *The dynamic effects of aggregate
demand and supply disturbances*. American Economic Review.

Calvo, G. A. and Reinhart, C. M. (2002). *Fear of floating*. Quarterly
Journal of Economics.

Mundell, R. A. (1961). *A theory of optimum currency areas*. American
Economic Review.

Full reference list: see `paper.pdf`.

---

## License & academic integrity

This repository is submitted as individual coursework for evaluation.
The econometric reasoning, code and writing are the author's own; AI
tools and published references were consulted as permitted by the
assignment's collaboration policy. Identical or near-identical
submissions would constitute plagiarism.

"""
================================================================================
Counterfactual Inflation Analysis: What If Ukraine Had Been Part of the Euro Area?
================================================================================
Final Exam — Quantitative Methods in Finance — Master 2 Research
Academic Year 2025-2026

METHODOLOGY:
- Bayoumi-Eichengreen (1993) bivariate Structural VAR
- Long-run identification à la Blanchard-Quah (1989):
    * Supply shocks: permanent effect on output
    * Demand shocks: NO permanent effect on output (transitory only)
- Counterfactual via SHOCK REPLACEMENT:
    Ukraine keeps its own SUPPLY shocks (energy, agriculture, geopolitics)
    Ukraine inherits Euro Area DEMAND shocks (ECB monetary policy)
- Time-varying treatment intensity theta_t based on Part A monetary regime analysis:
    Hard peg periods --> theta close to 0 (treatment small, EA membership ~ status quo)
    Floating / IT periods --> theta close to 1 (treatment large)
- Estimation sample: 2006-05 to 2021-12 (excluding the full-scale war)
- Projection sample: 2022-01 onwards using observed supply shocks

DATA:
- Data.xlsx (sheet 'monthly_data'): inflation YoY UA & EZ, industrial prod YoY UA & EZ,
  UAH/USD, UAH/EUR
- Data.xlsx (sheet 'data_ecb_hicp_panel'): HICP YoY for 11 EA countries (sanity check)
- Data.xlsx (sheet 'daily_data'): commodity prices (used to characterize supply shocks)

OUTPUT:
- output/fig_main_counterfactual.png : main figure (Part B deliverable)
- output/fig_irf.png                  : structural impulse response functions
- output/fig_shocks.png               : extracted structural shocks
- output/fig_uah_usd_partA.png        : exchange rate chronology (Part A)
- output/partA_regime_chronology.csv  : monetary regime table (Part A)
- output/counterfactual_series.csv    : actual vs counterfactual inflation series
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ============================================================
# 0. SETUP
# ============================================================
DATA_PATH = "Data.xlsx"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estimation sample (excluding the full-scale war shock 2022+)
EST_START = "2006-05-01"
EST_END   = "2021-12-31"
# Projection sample (full available range)
FULL_END  = "2025-04-30"

# Number of bootstrap replications for IRF confidence bands
N_BOOT = 500
np.random.seed(42)


# ============================================================
# 1. DATA LOADING & PREPARATION
# ============================================================
def load_data(path=DATA_PATH):
    """Load monthly data and HICP panel from the Excel file."""
    monthly = pd.read_excel(path, sheet_name="monthly_data", parse_dates=["Date"])
    monthly = monthly.sort_values("Date").reset_index(drop=True)
    monthly["Date"] = monthly["Date"].dt.to_period("M").dt.to_timestamp()  # normalize to month-start
    monthly = monthly.set_index("Date")

    hicp = pd.read_excel(path, sheet_name="data_ecb_hicp_panel", parse_dates=["Date"])
    hicp = hicp.sort_values("Date").reset_index(drop=True)
    hicp = hicp.set_index("Date")

    return monthly, hicp


def adf_report(series, name):
    """Quick ADF stationarity report (printed to console)."""
    s = series.dropna()
    res = adfuller(s, autolag="AIC")
    print(f"  ADF {name:<25s}: stat={res[0]:7.3f}, p-value={res[1]:6.3f}, "
          f"lags={res[2]}, n={res[3]}  --> "
          f"{'STATIONARY' if res[1] < 0.05 else 'non-stationary'}")
    return res[1]


# ============================================================
# 2. PART A: MONETARY REGIME CHRONOLOGY
# ============================================================
def part_A_regime_analysis(monthly):
    """
    Build chronology of NBU exchange rate regimes using UAH/USD movements
    and a measure of monetary sovereignty theta_t in [0, 1].
    """
    df = monthly.copy()
    # The UAH_USD column gives 1 UAH in USD -> invert to USD/UAH (more readable)
    df["USD_UAH"] = 1.0 / df["UAH_USD"]
    df["dlog_FX"] = np.log(df["USD_UAH"]).diff()
    # 12-month rolling stdev of monthly log changes -> proxy of FX flexibility
    df["FX_vol_12m"] = df["dlog_FX"].rolling(12).std()

    # Regime chronology (built on documented IMF / NBU sources, see references)
    # Sovereignty assignment reflects the COUNTERFACTUAL treatment intensity:
    #   "How much would EA membership have changed the inflation outcome?"
    # - During hard pegs to USD (2006-2008, 2010-2013), the de facto anchor
    #   was already a foreign currency, so EA membership would have been close
    #   to a re-anchoring (small treatment).
    # - During devaluation episodes (2008-09, 2014-15) and the IT regime,
    #   Ukraine actively used monetary autonomy: EA membership would have
    #   forbidden these adjustments (large treatment).
    # - The 2022-23 wartime peg is conceptually different: it was a forced
    #   defensive peg precisely because Ukraine lacks an international reserve
    #   currency. Under EA membership, Ukraine would have HAD the euro as
    #   reserve currency (no need to defend a peg, ECB as LOLR). So we treat
    #   the wartime peg as HIGH counterfactual sovereignty (large treatment).
    regimes = pd.DataFrame([
        ("2006-05", "2008-08", "Stabilised arrangement (de facto USD peg ~5.05)",
         "Quasi-fixed UAH/USD ~5.05; high reserves accumulation; strong FX market interventions",
         "Low"),
        ("2008-09", "2009-12", "Major devaluation (Global Financial Crisis)",
         "UAH collapse from ~5.0 to ~8.0/USD (-~50%); IMF Stand-By Arrangement",
         "High"),
        ("2010-01", "2013-12", "Re-pegged (de facto USD peg ~7.99)",
         "Restored quasi-fixed peg ~7.99 UAH/USD; capital controls; FX interventions",
         "Low"),
        ("2014-01", "2015-12", "Currency collapse (Crimea + Donbas)",
         "UAH collapsed from ~8 to ~25+/USD (-~200%); float adopted under IMF EFF program",
         "High"),
        ("2016-01", "2022-01", "Inflation Targeting (officially launched 2015-12)",
         "Managed float; IT regime; gradual monetary policy framework convergence",
         "High"),
        ("2022-02", "2023-09", "War-time defensive peg (full-scale invasion)",
         "Hryvnia fixed at 36.57/USD; capital controls re-imposed; absent EA membership "
         "Ukraine had no reserve currency to draw on, hence the costly peg",
         "High"),
        ("2023-10", "2025-04", "Managed flexibility",
         "Gradual return to managed float; stepwise depreciation",
         "High"),
    ], columns=["Period_start", "Period_end", "De_facto_regime", "Key_events", "Sovereignty"])

    regimes.to_csv(f"{OUTPUT_DIR}/partA_regime_chronology.csv", index=False)

    # Build theta_t (monetary sovereignty index) on the monthly frequency
    # 0 = hard peg (no sovereignty: monetary policy is subordinated to FX anchor)
    # 1 = full sovereignty (floating + IT)
    theta = pd.Series(index=df.index, dtype=float)
    map_sov = {"Low": 0.10, "Medium": 0.60, "High": 1.00}
    for _, r in regimes.iterrows():
        mask = (df.index >= r["Period_start"]) & (df.index <= r["Period_end"])
        theta.loc[mask] = map_sov[r["Sovereignty"]]
    theta = theta.ffill().fillna(map_sov["Low"])

    return df, regimes, theta


def plot_partA(df, regimes):
    """Plot UAH/USD with regime annotations."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["USD_UAH"], color="#1f4e79", lw=1.6, label="UAH per USD")

    # Shade regimes
    colors = {
        "Low":    "#cce5cc",
        "Medium": "#fff0b3",
        "High":   "#fbd0c8",
    }
    for _, r in regimes.iterrows():
        ax.axvspan(pd.Timestamp(r["Period_start"]), pd.Timestamp(r["Period_end"]),
                   color=colors[r["Sovereignty"]], alpha=0.45, lw=0)

    # Annotate the three big devaluations
    ax.annotate("GFC devaluation\n2008–09", xy=(pd.Timestamp("2009-02-01"), 8),
                xytext=(pd.Timestamp("2007-01-01"), 18),
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)
    ax.annotate("Crimea / Donbas\ncollapse 2014–15", xy=(pd.Timestamp("2015-02-01"), 27),
                xytext=(pd.Timestamp("2011-01-01"), 32),
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)
    ax.annotate("Full-scale\ninvasion 2022", xy=(pd.Timestamp("2022-04-01"), 30),
                xytext=(pd.Timestamp("2017-06-01"), 38),
                arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)

    # Legend for regimes
    handles = [
        Rectangle((0, 0), 1, 1, color=colors["Low"], alpha=0.6, label="Low sovereignty (peg)"),
        Rectangle((0, 0), 1, 1, color=colors["Medium"], alpha=0.6, label="Medium (managed float)"),
        Rectangle((0, 0), 1, 1, color=colors["High"], alpha=0.6, label="High sovereignty (float / IT)"),
    ]
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="#1f4e79", lw=1.6, label="UAH/USD")],
              loc="upper left", framealpha=0.9)
    ax.set_title("Part A — Ukraine Exchange Rate Regimes (UAH per USD), 2006–2025",
                 pad=12)
    ax.set_ylabel("UAH per USD")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 48)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_uah_usd_partA.png", dpi=140)
    plt.close()


# ============================================================
# 3. PART B: SVAR ESTIMATION (BAYOUMI-EICHENGREEN)
# ============================================================
def estimate_svar(y, max_lag=6, label=""):
    """
    Estimate a bivariate VAR on y = [d_output, inflation], select lag length,
    and recover Blanchard-Quah long-run identification.

    Returns:
        var_res     : statsmodels VAR results object
        p           : selected lag order
        B           : 2x2 structural impact matrix (B = C(1)^-1 * L)
                      e_t = B * eps_t  with eps_t = (eps_supply, eps_demand)
        eps         : extracted structural shocks (T-p, 2) array
        IRF_struct  : structural IRF, array (h+1, 2, 2)
                      IRF_struct[h, i, j] = response of variable i at horizon h
                      to a unit structural shock j (j=0 supply, j=1 demand)
    """
    y = y.dropna().values
    var_model = VAR(y)
    # Lag selection
    sel = var_model.select_order(maxlags=max_lag)
    # Use AIC; fall back to a sensible default
    p = max(1, int(sel.aic)) if sel.aic else 2
    print(f"  [{label}] Lag selection (AIC): p = {p}")
    var_res = var_model.fit(p)

    # Reduced-form residuals covariance
    Sigma = np.cov(var_res.resid.T, ddof=var_res.df_resid > 0)

    # Long-run multiplier C(1) = (I - A1 - A2 - ... - Ap)^{-1}
    A_sum = np.zeros((2, 2))
    for i in range(p):
        A_sum += var_res.coefs[i]
    C1 = np.linalg.inv(np.eye(2) - A_sum)

    # Long-run covariance: C(1) Sigma C(1)'
    LR = C1 @ Sigma @ C1.T
    # Cholesky -> lower triangular L such that L L' = LR
    L = np.linalg.cholesky(LR)
    # Long-run identification: C(1) B = L  =>  B = C(1)^{-1} L
    B = np.linalg.solve(C1, L)

    # Sign convention:
    #  - column 0 (supply) should have a NEGATIVE long-run impact on inflation
    #    (positive supply shock lowers prices) => check sign of L[1, 0]
    #    L[1,0] is the cumulated long-run effect of supply on inflation; force it negative.
    if L[1, 0] > 0:
        B[:, 0] *= -1
    #  - column 1 (demand) raises inflation in the long run; L[1,1] > 0
    if L[1, 1] < 0:
        B[:, 1] *= -1

    # Extract structural shocks: eps_t = B^{-1} e_t
    B_inv = np.linalg.inv(B)
    eps = (var_res.resid @ B_inv.T)  # shape (T-p, 2)

    # Structural IRFs (horizon up to 60 months)
    H = 60
    irf_red = var_res.irf(H).irfs           # (H+1, K, K) reduced-form IRF (orthogonalize ourselves)
    # Build IRF: response = irf_red[h] @ B
    IRF_struct = np.zeros((H + 1, 2, 2))
    for h in range(H + 1):
        IRF_struct[h] = irf_red[h] @ B  # impact of structural shocks

    return var_res, p, B, eps, IRF_struct, Sigma, C1


def historical_decomposition(var_res, B, eps, y_full):
    """
    Build the historical contribution of structural shocks to inflation level.
    For variable i, time t > p:
        y_i,t = deterministic + sum_{h=0..t-p-1} sum_{j} IRF_struct[h, i, j] * eps_{t-h, j}
    We reconstruct only the stochastic part (innovation contributions), then add
    the unconditional mean of inflation as the constant baseline.

    Returns (T, 2) array: contribution of supply (col 0) and demand (col 1)
    to the inflation series (variable index 1).
    """
    p = var_res.k_ar
    T = len(y_full)
    # Reduced-form IRFs over the full sample horizon
    H = T - p - 1
    irf_red = var_res.irf(H).irfs  # (H+1, 2, 2)
    # Pre-multiply by B to get structural IRFs
    IRF_s = np.zeros_like(irf_red)
    for h in range(H + 1):
        IRF_s[h] = irf_red[h] @ B

    # Contribution of each structural shock to inflation (variable index 1)
    contrib = np.zeros((T, 2))  # (time, [supply, demand])
    for t in range(p, T):
        h_max = t - p
        # eps[s] corresponds to time index p+s (s=0..h_max)
        for h in range(h_max + 1):
            contrib[t, 0] += IRF_s[h, 1, 0] * eps[h_max - h, 0]
            contrib[t, 1] += IRF_s[h, 1, 1] * eps[h_max - h, 1]

    return contrib


def bootstrap_irf_bands(y, p, B_baseline, n_boot=N_BOOT, H=60, conf=0.68):
    """
    Bootstrap confidence bands for the structural IRF on inflation (variable 1).
    Residual-bootstrap using the estimated reduced-form VAR.
    Returns lower and upper IRF arrays of shape (H+1, 2).
    """
    y = y.dropna().values
    var_res = VAR(y).fit(p)
    resid = var_res.resid          # (T-p, 2)
    intercept = var_res.intercept  # (2,) or (k,)
    coefs = var_res.coefs          # (p, 2, 2)
    T = len(y)
    K = 2

    boots = np.zeros((n_boot, H + 1, 2))  # store inflation responses (col 1)

    for b in range(n_boot):
        # Sample residuals
        idx = np.random.randint(0, len(resid), size=len(resid))
        u_b = resid[idx]
        # Simulate y* from the same starting values
        y_b = y.copy()
        for t in range(p, T):
            xt = intercept.copy() if intercept is not None else np.zeros(K)
            for lag in range(p):
                xt = xt + coefs[lag] @ y_b[t - lag - 1]
            y_b[t] = xt + u_b[t - p]
        # Re-estimate VAR(p) on the bootstrapped sample
        try:
            res_b = VAR(y_b).fit(p)
            Sigma_b = np.cov(res_b.resid.T, ddof=res_b.df_resid > 0)
            A_sum = np.zeros((K, K))
            for i in range(p):
                A_sum += res_b.coefs[i]
            C1_b = np.linalg.inv(np.eye(K) - A_sum)
            LR_b = C1_b @ Sigma_b @ C1_b.T
            L_b = np.linalg.cholesky(LR_b)
            B_b = np.linalg.solve(C1_b, L_b)
            # Sign convention (consistent with baseline)
            if L_b[1, 0] > 0:
                B_b[:, 0] *= -1
            if L_b[1, 1] < 0:
                B_b[:, 1] *= -1
            irf_red_b = res_b.irf(H).irfs
            for h in range(H + 1):
                Mh = irf_red_b[h] @ B_b
                boots[b, h, 0] = Mh[1, 0]   # inflation response to supply
                boots[b, h, 1] = Mh[1, 1]   # inflation response to demand
        except Exception:
            boots[b] = np.nan

    alpha = (1 - conf) / 2
    lower = np.nanpercentile(boots, 100 * alpha, axis=0)
    upper = np.nanpercentile(boots, 100 * (1 - alpha), axis=0)
    return lower, upper


# ============================================================
# 4. PART B: COUNTERFACTUAL CONSTRUCTION
# ============================================================
def build_counterfactual(monthly, theta, y_ua_full, y_ez_full,
                          var_ua, B_ua, eps_ua_full,
                          var_ez, B_ez, eps_ez_full):
    """
    Counterfactual construction via shock replacement.

    KEY IDEA. Under Euro Area membership:
      (1) Monetary policy is set by the ECB for the whole area, not by the NBU.
          Hence, Ukraine's idiosyncratic DEMAND shocks (driven by national
          monetary/fiscal policy and credibility) are NOT inherited; instead,
          Ukraine's inflation is anchored on the contemporaneous Euro Area
          inflation rate pi_EZ_t (which already embeds ECB monetary policy
          and the area-wide demand shocks).
      (2) Ukraine remains exposed to its OWN structural SUPPLY shocks
          (energy dependence, agricultural commodities, geopolitical
          disruptions). These are real, idiosyncratic and persist under
          any monetary regime.

    Counterfactual inflation:
        pi_cf_t = pi_EZ_t  +  sum_{h=0..H} psi^UA_supply,h * eps_supply^UA_{t-h}
                           +  alpha * sum_{h=0..H} psi^EZ_demand,h * (eps^EZ_d_t - eps^EZ_d,baseline)

    where the last term captures any deviation of Euro Area demand conditions
    relative to the area baseline (small correction; the bulk of the demand
    contribution is already in pi_EZ_t).

    For simplicity and following Bayoumi-Eichengreen's identification logic,
    we use the parsimonious form:
        pi_cf_t = pi_EZ_t + supply_contribution_UA(t)

    Then we apply the time-varying treatment theta_t (Part A):
      pi_cf_weighted_t = (1 - theta_t) * pi_actual_t + theta_t * pi_cf_pure_t
    During hard-peg periods (theta close to 0), Ukraine's monetary policy
    was already subordinated to a foreign anchor (USD), so EA membership
    would have brought only a marginal change.
    """
    p_ua = var_ua.k_ar
    T = len(y_ua_full)

    # Structural IRFs for Ukraine, full horizon
    H = T - p_ua - 1
    irf_red_ua = var_ua.irf(H).irfs
    IRF_s_ua = np.zeros_like(irf_red_ua)
    for h in range(H + 1):
        IRF_s_ua[h] = irf_red_ua[h] @ B_ua

    # Align shock series in calendar time
    p_ez = var_ez.k_ar
    p_max = max(p_ua, p_ez)
    offset_ua = p_max - p_ua
    offset_ez = p_max - p_ez
    eps_ua_aligned = eps_ua_full[offset_ua:]
    eps_ez_aligned = eps_ez_full[offset_ez:]
    n_common = min(len(eps_ua_aligned), len(eps_ez_aligned))
    eps_ua_aligned = eps_ua_aligned[:n_common]
    eps_ez_aligned = eps_ez_aligned[:n_common]

    dates_full = monthly.index
    dates_shocks = dates_full[p_max:p_max + n_common]

    # --- Supply contribution for Ukraine (using its own IRFs and own supply shocks) ---
    supply_contrib_ua = np.zeros(n_common)
    for t in range(n_common):
        for h in range(t + 1):
            supply_contrib_ua[t] += IRF_s_ua[h, 1, 0] * eps_ua_aligned[t - h, 0]

    # --- Anchor: contemporaneous Euro Area inflation (imports ECB credibility) ---
    pi_ez_obs = monthly["Inflation_Eurozone"].reindex(dates_shocks).values

    # --- PURE counterfactual (full-treatment, theta = 1) ---
    pi_cf_pure = pi_ez_obs + supply_contrib_ua

    # --- Sanity series: Ukraine reconstructed with ITS OWN demand AND supply shocks ---
    # This should approximately equal observed Ukraine inflation if the SVAR fits well.
    demand_contrib_ua = np.zeros(n_common)
    for t in range(n_common):
        for h in range(t + 1):
            demand_contrib_ua[t] += IRF_s_ua[h, 1, 1] * eps_ua_aligned[t - h, 1]
    est_mask = (monthly.index >= EST_START) & (monthly.index <= EST_END)
    mu_ua = monthly.loc[est_mask, "Inflation_Ukraine"].mean()
    pi_actual_recons = mu_ua + supply_contrib_ua + demand_contrib_ua

    # Build pandas Series on the full monthly index
    cf_pure = pd.Series(np.nan, index=dates_full, name="pi_cf_pure")
    cf_pure.loc[dates_shocks] = pi_cf_pure
    sanity = pd.Series(np.nan, index=dates_full, name="pi_actual_reconstructed")
    sanity.loc[dates_shocks] = pi_actual_recons

    # --- Theta-weighted counterfactual ---
    pi_obs_series = monthly["Inflation_Ukraine"]
    cf_weighted = (1.0 - theta) * pi_obs_series + theta * cf_pure

    return cf_pure, cf_weighted, sanity, dates_shocks, mu_ua, None


# ============================================================
# 5. PLOTTING (Part B figures)
# ============================================================
def plot_main_counterfactual(monthly, cf_pure, cf_weighted, theta):
    """Main deliverable: actual vs counterfactual inflation, with theta panel below."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                  gridspec_kw={"height_ratios": [3.5, 1]},
                                  sharex=True)

    # Crisis bands
    crises = [
        ("2008-08", "2009-12", "GFC"),
        ("2014-01", "2015-12", "Crimea / Donbas"),
        ("2022-02", "2023-09", "Full-scale invasion"),
    ]
    for (s, e, lbl) in crises:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.13, lw=0)
        ax2.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.13, lw=0)
        ax.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, 64, lbl,
                ha="center", va="top", fontsize=9, color="dimgrey")

    ax.plot(monthly.index, monthly["Inflation_Ukraine"],
            color="#a02020", lw=1.8, label="Actual Ukraine inflation (YoY %)")
    ax.plot(cf_weighted.index, cf_weighted.values,
            color="#1f4e79", lw=1.8,
            label=r"Counterfactual: Ukraine in the Euro Area (weighted by $\theta_t$)")
    ax.plot(cf_pure.index, cf_pure.values,
            color="#1f4e79", lw=1.0, ls="--", alpha=0.55,
            label=r"Counterfactual (pure SVAR shock-replacement, $\theta_t = 1$)")
    ax.plot(monthly.index, monthly["Inflation_Eurozone"],
            color="#2e7d32", lw=1.2, alpha=0.85,
            label="Euro Area average inflation (YoY %, reference)")

    ax.axhline(2.0, color="#2e7d32", ls=":", lw=0.8, alpha=0.7)
    ax.text(pd.Timestamp("2006-08-01"), 2.4, "ECB target = 2%", color="#2e7d32", fontsize=8)

    ax.set_title("Counterfactual Inflation: What If Ukraine Had Been in the Euro Area?\n"
                 "Bayoumi–Eichengreen SVAR with Blanchard–Quah long-run identification",
                 fontsize=12)
    ax.set_ylabel("Inflation, YoY %")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-6, 68)

    # Theta panel
    ax2.fill_between(theta.index, theta.values, color="#1f4e79", alpha=0.30)
    ax2.plot(theta.index, theta.values, color="#1f4e79", lw=1.4)
    ax2.set_ylabel(r"$\theta_t$" + "\n(monetary\nsovereignty)", fontsize=9)
    ax2.set_xlabel("Date")
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_main_counterfactual.png", dpi=140)
    plt.close()


def plot_irf(IRF_ua, IRF_ez, irf_lo_ua, irf_hi_ua, irf_lo_ez, irf_hi_ez):
    """Plot impulse response functions of inflation to supply / demand shocks."""
    H = IRF_ua.shape[0]
    horizons = np.arange(H)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    # Ukraine: supply shock -> inflation
    axes[0, 0].fill_between(horizons, irf_lo_ua[:, 0], irf_hi_ua[:, 0],
                             color="#a02020", alpha=0.20, label="68% CI (bootstrap)")
    axes[0, 0].plot(horizons, IRF_ua[:, 1, 0], color="#a02020", lw=2)
    axes[0, 0].axhline(0, color="black", lw=0.5)
    axes[0, 0].set_title("Ukraine: response of inflation to a SUPPLY shock")
    axes[0, 0].set_ylabel("YoY inflation, pp")
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Ukraine: demand shock -> inflation
    axes[0, 1].fill_between(horizons, irf_lo_ua[:, 1], irf_hi_ua[:, 1],
                             color="#a02020", alpha=0.20)
    axes[0, 1].plot(horizons, IRF_ua[:, 1, 1], color="#a02020", lw=2)
    axes[0, 1].axhline(0, color="black", lw=0.5)
    axes[0, 1].set_title("Ukraine: response of inflation to a DEMAND shock")
    axes[0, 1].grid(alpha=0.3)

    # Euro Area: supply shock -> inflation
    axes[1, 0].fill_between(horizons, irf_lo_ez[:, 0], irf_hi_ez[:, 0],
                             color="#1f4e79", alpha=0.20, label="68% CI (bootstrap)")
    axes[1, 0].plot(horizons, IRF_ez[:, 1, 0], color="#1f4e79", lw=2)
    axes[1, 0].axhline(0, color="black", lw=0.5)
    axes[1, 0].set_title("Euro Area: response of inflation to a SUPPLY shock")
    axes[1, 0].set_xlabel("Horizon (months)")
    axes[1, 0].set_ylabel("YoY inflation, pp")
    axes[1, 0].legend(loc="best", fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Euro Area: demand shock -> inflation
    axes[1, 1].fill_between(horizons, irf_lo_ez[:, 1], irf_hi_ez[:, 1],
                             color="#1f4e79", alpha=0.20)
    axes[1, 1].plot(horizons, IRF_ez[:, 1, 1], color="#1f4e79", lw=2)
    axes[1, 1].axhline(0, color="black", lw=0.5)
    axes[1, 1].set_title("Euro Area: response of inflation to a DEMAND shock")
    axes[1, 1].set_xlabel("Horizon (months)")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Structural impulse response functions — Blanchard–Quah identification",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_irf.png", dpi=140)
    plt.close()


def plot_shocks(dates_shocks_ua, eps_ua, dates_shocks_ez, eps_ez):
    """Plot the extracted structural shocks for both economies."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    axes[0, 0].plot(dates_shocks_ua, eps_ua[:, 0], color="#a02020", lw=0.9)
    axes[0, 0].axhline(0, color="black", lw=0.5)
    axes[0, 0].set_title("Ukraine: SUPPLY shocks")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(dates_shocks_ua, eps_ua[:, 1], color="#a02020", lw=0.9)
    axes[0, 1].axhline(0, color="black", lw=0.5)
    axes[0, 1].set_title("Ukraine: DEMAND shocks")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(dates_shocks_ez, eps_ez[:, 0], color="#1f4e79", lw=0.9)
    axes[1, 0].axhline(0, color="black", lw=0.5)
    axes[1, 0].set_title("Euro Area: SUPPLY shocks")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(dates_shocks_ez, eps_ez[:, 1], color="#1f4e79", lw=0.9)
    axes[1, 1].axhline(0, color="black", lw=0.5)
    axes[1, 1].set_title("Euro Area: DEMAND shocks")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Extracted structural shocks (Blanchard–Quah identification)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_shocks.png", dpi=140)
    plt.close()


# ============================================================
# 6. MAIN PIPELINE
# ============================================================
def main():
    print("=" * 78)
    print("Counterfactual Inflation Analysis: Ukraine in the Euro Area")
    print("=" * 78)

    # 1) Load data ------------------------------------------------------------
    print("\n[1] Loading data ...")
    monthly, hicp = load_data()
    print(f"    monthly_data shape: {monthly.shape}, range: {monthly.index.min()} -> {monthly.index.max()}")
    print(f"    hicp panel  shape: {hicp.shape}, range: {hicp.index.min()} -> {hicp.index.max()}")

    # 2) Stationarity tests ---------------------------------------------------
    print("\n[2] Stationarity (ADF) tests on full sample:")
    adf_report(monthly["Inflation_Ukraine"], "Inflation_Ukraine")
    adf_report(monthly["Inflation_Eurozone"], "Inflation_Eurozone")
    adf_report(monthly["IndustrialProd_Ukraine"], "IndProd_Ukraine (YoY)")
    adf_report(monthly["IndustrialProd_Eurozone"], "IndProd_Eurozone (YoY)")
    # YoY series are typically stationary; we use them as-is.

    # 3) Part A : monetary regime chronology ---------------------------------
    print("\n[3] Part A — building monetary regime chronology and theta_t ...")
    df_partA, regimes, theta = part_A_regime_analysis(monthly)
    plot_partA(df_partA, regimes)
    print(f"    Saved: {OUTPUT_DIR}/fig_uah_usd_partA.png")
    print(f"    Saved: {OUTPUT_DIR}/partA_regime_chronology.csv")
    print(f"    theta_t summary: mean={theta.mean():.2f}, "
          f"share with theta=1: {(theta == 1.0).mean():.0%}, "
          f"share with theta<=0.1: {(theta <= 0.10).mean():.0%}")

    # 4) Build estimation samples (peace, pre-war) ---------------------------
    print("\n[4] Estimating SVAR on PEACE sample (2006-05 to 2021-12) ...")
    est_mask = (monthly.index >= EST_START) & (monthly.index <= EST_END)
    full_mask = (monthly.index <= FULL_END)

    y_ua_est = monthly.loc[est_mask, ["IndustrialProd_Ukraine", "Inflation_Ukraine"]]
    y_ez_est = monthly.loc[est_mask, ["IndustrialProd_Eurozone", "Inflation_Eurozone"]]
    y_ua_full = monthly.loc[full_mask, ["IndustrialProd_Ukraine", "Inflation_Ukraine"]]
    y_ez_full = monthly.loc[full_mask, ["IndustrialProd_Eurozone", "Inflation_Eurozone"]]

    # Demean on estimation sample for SVAR (BQ requires zero-mean)
    mu_ua = y_ua_est.mean()
    mu_ez = y_ez_est.mean()
    y_ua_est_dm = y_ua_est - mu_ua
    y_ez_est_dm = y_ez_est - mu_ez

    # 5) SVAR Ukraine ---------------------------------------------------------
    print("\n[5] SVAR Ukraine ...")
    var_ua, p_ua, B_ua, eps_ua_est, IRF_ua, Sigma_ua, C1_ua = estimate_svar(
        y_ua_est_dm, max_lag=6, label="Ukraine"
    )
    print(f"    B_ua =\n{B_ua}")
    print(f"    Long-run effects on inflation: supply={C1_ua[1, 0]*B_ua[0,0]+C1_ua[1,1]*B_ua[1,0]:.3f}, "
          f"demand={C1_ua[1, 0]*B_ua[0,1]+C1_ua[1,1]*B_ua[1,1]:.3f}")

    # SVAR Euro Area
    print("\n[6] SVAR Euro Area ...")
    var_ez, p_ez, B_ez, eps_ez_est, IRF_ez, Sigma_ez, C1_ez = estimate_svar(
        y_ez_est_dm, max_lag=6, label="EuroArea"
    )

    # 6) Bootstrap confidence bands on inflation IRFs ------------------------
    print(f"\n[7] Bootstrapping IRF confidence bands ({N_BOOT} replications) ...")
    irf_lo_ua, irf_hi_ua = bootstrap_irf_bands(y_ua_est_dm, p_ua, B_ua, n_boot=N_BOOT, H=60)
    irf_lo_ez, irf_hi_ez = bootstrap_irf_bands(y_ez_est_dm, p_ez, B_ez, n_boot=N_BOOT, H=60)
    plot_irf(IRF_ua, IRF_ez, irf_lo_ua, irf_hi_ua, irf_lo_ez, irf_hi_ez)
    print(f"    Saved: {OUTPUT_DIR}/fig_irf.png")

    # 7) Extract structural shocks on the FULL sample using the estimated SVAR
    # We refit the reduced-form VAR on the full sample (with demeaning by the
    # peace-sample mean) to obtain residuals, then apply the SAME B matrices.
    print("\n[8] Extracting structural shocks on full sample (incl. war) ...")

    def fit_full_resid(y_full, mu, p):
        y_dm = (y_full - mu).values
        var_full = VAR(y_dm).fit(p)
        return var_full

    var_ua_full = fit_full_resid(y_ua_full, mu_ua, p_ua)
    var_ez_full = fit_full_resid(y_ez_full, mu_ez, p_ez)

    eps_ua_full = (var_ua_full.resid @ np.linalg.inv(B_ua).T)
    eps_ez_full = (var_ez_full.resid @ np.linalg.inv(B_ez).T)

    # Dates corresponding to the shock series
    dates_ua_shocks = y_ua_full.index[p_ua:]
    dates_ez_shocks = y_ez_full.index[p_ez:]

    # 8) Counterfactual via shock replacement --------------------------------
    print("\n[9] Building the counterfactual (shock replacement + theta_t weighting) ...")
    cf_pure, cf_weighted, sanity, dates_shocks, _, _ = build_counterfactual(
        monthly.loc[full_mask], theta.loc[full_mask],
        y_ua_full, y_ez_full,
        var_ua_full, B_ua, eps_ua_full,
        var_ez_full, B_ez, eps_ez_full
    )

    # 9) Plots ---------------------------------------------------------------
    print("\n[10] Producing figures ...")
    plot_shocks(dates_ua_shocks, eps_ua_full, dates_ez_shocks, eps_ez_full)
    print(f"    Saved: {OUTPUT_DIR}/fig_shocks.png")
    plot_main_counterfactual(monthly.loc[full_mask], cf_pure, cf_weighted, theta.loc[full_mask])
    print(f"    Saved: {OUTPUT_DIR}/fig_main_counterfactual.png")

    # 10) Export final series ------------------------------------------------
    out = pd.DataFrame({
        "Inflation_Ukraine_actual":   monthly["Inflation_Ukraine"],
        "Inflation_Eurozone_actual":  monthly["Inflation_Eurozone"],
        "Counterfactual_pure":        cf_pure,
        "Counterfactual_weighted":    cf_weighted,
        "theta_sovereignty":          theta,
    })
    out.to_csv(f"{OUTPUT_DIR}/counterfactual_series.csv", index_label="Date")
    print(f"    Saved: {OUTPUT_DIR}/counterfactual_series.csv")

    # 11) Quantitative summary -----------------------------------------------
    print("\n[11] Quantitative summary of the counterfactual:")
    diff = (monthly["Inflation_Ukraine"] - cf_weighted).dropna()
    print(f"    Mean gap (actual - CF, weighted): {diff.mean():.2f} pp")
    print(f"    Cumulative pre-2016 gap (peg era): "
          f"{diff.loc[diff.index < '2016-01-01'].mean():.2f} pp")
    print(f"    Cumulative 2016-2022 gap (IT):    "
          f"{diff.loc[(diff.index >= '2016-01-01') & (diff.index < '2022-01-01')].mean():.2f} pp")
    print(f"    War period gap (2022-2025):       "
          f"{diff.loc[diff.index >= '2022-01-01'].mean():.2f} pp")

    print("\n" + "=" * 78)
    print("DONE — all outputs written to ./output/")
    print("=" * 78)


if __name__ == "__main__":
    main()

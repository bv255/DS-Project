"""
Synthetic Regime Data Generator (Function Version)
====================================================
Generates synthetic market data with KNOWN regime labels for validating
regime detection models (HMM, GMM, Changepoint).

Methodology adapted from Bucci & Ciciretti (2022), "Market regime detection
via realized covariances", Economic Modelling, 111, 105832.

Following their first-pillar validation approach, we use an AR(1) process
to generate a smooth, persistent volatility driver that controls regime
switches. The mean is held stable while the variance is modified across
regimes, allowing us to identify exactly which periods belong to a highly
volatile (bear) regime and which to a calm (bull) regime.

Usage:
    from synthetic_regime_generator import generate_synthetic_regime_data

    # Default parameters
    df = generate_synthetic_regime_data()

    # Sensitivity analysis example
    df = generate_synthetic_regime_data(ar_coeff=0.95, bear_std_return=0.015)
"""

import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


def generate_synthetic_regime_data(
    # Core parameters
    T=2500,
    seed=42,

    # Latent stress AR(1)
    ar_coeff=0.995,
    ar_noise_std=0.08,

    # Hysteresis thresholds on composite stress score
    bear_entry_percentile=80,
    bear_exit_percentile=65,
    min_regime_duration=20,

    # Return dynamics at low vs high stress
    bull_mean_return=0.0006,
    bear_mean_return=-0.0012,
    bull_std_return=0.007,
    bear_std_return=0.022,

    # Synthetic VIX at low vs high stress
    bull_vix_mean=14.0,
    bear_vix_mean=32.0,
    bull_vix_std=2.0,
    bear_vix_std=5.0,

    # Synthetic sentiment at low vs high stress
    bull_sentiment_mean=85.0,
    bear_sentiment_mean=65.0,
    bull_sentiment_std=7.0,
    bear_sentiment_std=10.0,

    # Oil dynamics
    bull_oil_drift=0.0003,
    bear_oil_drift=-0.0008,
    bull_oil_std=0.010,
    bear_oil_std=0.018,

    # Stress score weights
    w_stress=0.55,
    w_drawdown=0.30,
    w_negret=0.15,

    # Windows
    ret_window=21,
    sentiment_smooth_window=21,
    sentiment_z_window=252,
    sentiment_z_min_periods=63,

    # Price
    start_price=1000.0,
):
    """
    Generate synthetic market data with known bull/bear regimes using a
    model-agnostic latent stress process.

    Design:
    1. Generate a persistent latent AR(1) stress variable.
    2. Let stress continuously affect return drift and volatility.
    3. Build price path and derived market-pressure variables.
    4. Form a composite stress score from:
       - latent stress
       - drawdown pressure
       - negative rolling return pressure
    5. Apply hysteresis + minimum duration to obtain ground-truth regimes.

    This is intended to be fairer across HMM, GMM, and PELT than generating
    labels from a hidden Markov chain.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    # 1. Persistent latent stress process
    # ---------------------------------------------------------------
    latent = np.zeros(T)
    latent[0] = rng.normal(0.0, ar_noise_std)
    for t in range(1, T):
        latent[t] = ar_coeff * latent[t - 1] + rng.normal(0.0, ar_noise_std)

    # Scale to [0, 1]
    latent_min = latent.min()
    latent_max = latent.max()
    stress = (latent - latent_min) / (latent_max - latent_min + 1e-12)

    # ---------------------------------------------------------------
    # 2. Stress-dependent returns
    #    Higher stress => lower drift, higher volatility
    # ---------------------------------------------------------------
    mu_t = bull_mean_return + (bear_mean_return - bull_mean_return) * stress
    sigma_t = bull_std_return + (bear_std_return - bull_std_return) * stress
    returns = rng.normal(mu_t, sigma_t)

    # ---------------------------------------------------------------
    # 3. Price and market-pressure quantities
    # ---------------------------------------------------------------
    price = np.zeros(T)
    price[0] = start_price
    for t in range(1, T):
        price[t] = price[t - 1] * (1.0 + returns[t])

    cummax = np.maximum.accumulate(price)
    drawdown = (price - cummax) / cummax  # <= 0

    rolling_ret = pd.Series(returns).rolling(ret_window, min_periods=1).mean().values

    # Convert drawdown and rolling returns into [0,1] "pressure" measures
    # 20% drawdown ~= max stress; -2% avg daily rolling return ~= max stress
    drawdown_pressure = np.clip((-drawdown) / 0.20, 0.0, 1.0)
    neg_return_pressure = np.clip((-rolling_ret) / 0.02, 0.0, 1.0)

    # ---------------------------------------------------------------
    # 4. Composite stress score for regime labelling
    # ---------------------------------------------------------------
    stress_score = (
        w_stress * stress
        + w_drawdown * drawdown_pressure
        + w_negret * neg_return_pressure
    )

    entry_thr = np.percentile(stress_score, bear_entry_percentile)
    exit_thr = np.percentile(stress_score, bear_exit_percentile)

    # Hysteresis labelling
    true_regime = np.empty(T, dtype=object)
    true_regime[0] = "bull"
    for t in range(1, T):
        if true_regime[t - 1] == "bull" and stress_score[t] >= entry_thr:
            true_regime[t] = "bear"
        elif true_regime[t - 1] == "bear" and stress_score[t] <= exit_thr:
            true_regime[t] = "bull"
        else:
            true_regime[t] = true_regime[t - 1]

    # ---------------------------------------------------------------
    # 5. Enforce minimum regime duration
    # ---------------------------------------------------------------
    def _get_runs(labels):
        runs = []
        current = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current:
                runs.append((start, i, current))
                start = i
                current = labels[i]
        runs.append((start, len(labels), current))
        return runs

    regime_runs = _get_runs(true_regime)

    changed = True
    while changed:
        changed = False
        new_runs = []
        for i, (s, e, r) in enumerate(regime_runs):
            duration = e - s
            if duration < min_regime_duration and len(new_runs) > 0:
                prev_s, prev_e, prev_r = new_runs[-1]
                new_runs[-1] = (prev_s, e, prev_r)
                changed = True
            else:
                new_runs.append((s, e, r))
        regime_runs = new_runs

    true_regime = np.empty(T, dtype=object)
    for s, e, r in regime_runs:
        true_regime[s:e] = r

    # ---------------------------------------------------------------
    # 6. Stress-driven observed features (continuous, not discrete)
    # ---------------------------------------------------------------
    vix_mu = bull_vix_mean + (bear_vix_mean - bull_vix_mean) * stress
    vix_sigma = bull_vix_std + (bear_vix_std - bull_vix_std) * stress
    vix_noise = rng.normal(vix_mu, vix_sigma)
    
    # Smooth VIX with AR(1) to prevent unrealistic day-to-day flickering
    vix = np.zeros(T)
    vix[0] = vix_noise[0]
    vix_persistence = 0.92  # higher = smoother
    for t in range(1, T):
        vix[t] = vix_persistence * vix[t-1] + (1 - vix_persistence) * vix_noise[t]
    vix = np.clip(vix, 9, 80)

    sent_mu = bull_sentiment_mean + (bear_sentiment_mean - bull_sentiment_mean) * stress
    sent_sigma = bull_sentiment_std + (bear_sentiment_std - bull_sentiment_std) * stress
    sent_noise = rng.normal(sent_mu, sent_sigma)
    
    sentiment_raw = np.zeros(T)
    sentiment_raw[0] = sent_noise[0]
    sent_persistence = 0.95
    for t in range(1, T):
        sentiment_raw[t] = sent_persistence * sentiment_raw[t-1] + (1 - sent_persistence) * sent_noise[t]

    oil = np.zeros(T)
    oil[0] = 70.0
    oil_drift = bull_oil_drift + (bear_oil_drift - bull_oil_drift) * stress
    oil_vol = bull_oil_std + (bear_oil_std - bull_oil_std) * stress
    for t in range(1, T):
        oil[t] = oil[t - 1] * (1.0 + rng.normal(oil_drift[t], oil_vol[t]))

    # ---------------------------------------------------------------
    # 7. Technical features
    # ---------------------------------------------------------------
    # RSI_14
    rsi_14 = np.full(T, np.nan)
    for t in range(14, T):
        window_rets = returns[t - 14:t]
        gains = np.where(window_rets > 0, window_rets, 0.0)
        losses = np.where(window_rets < 0, -window_rets, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            rsi_14[t] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14[t] = 100.0 - (100.0 / (1.0 + rs))

    # MACD histogram
    def _ema(series, span):
        out = np.zeros(len(series))
        alpha = 2.0 / (span + 1.0)
        out[0] = series[0]
        for i in range(1, len(series)):
            out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
        return out

    ema_12 = _ema(price, 12)
    ema_26 = _ema(price, 26)
    macd_line = ema_12 - ema_26
    signal_line = _ema(macd_line, 9)
    macd_hist = macd_line - signal_line

    # Consumer sentiment z-score
    sentiment_smooth = (
        pd.Series(sentiment_raw)
        .rolling(sentiment_smooth_window, min_periods=1)
        .mean()
        .values
    )

    rolling_mean_sent = pd.Series(sentiment_smooth).rolling(
        sentiment_z_window, min_periods=sentiment_z_min_periods
    ).mean()

    rolling_std_sent = pd.Series(sentiment_smooth).rolling(
        sentiment_z_window, min_periods=sentiment_z_min_periods
    ).std()

    consumer_sentiment_zscore = (
        (sentiment_smooth - rolling_mean_sent) / rolling_std_sent
    ).values

    # ---------------------------------------------------------------
    # 8. Assemble dataframe
    # ---------------------------------------------------------------
    dates = pd.bdate_range(start="2010-01-04", periods=T, freq="B")

    df = pd.DataFrame({
        "Date": dates,
        "GSPC": price,
        "Return": returns,
        "VIX": vix,
        "Drawdown": drawdown,
        "RSI_14": rsi_14,
        "MACD_Hist": macd_hist,
        "Consumer_Sentiment_ZScore": consumer_sentiment_zscore,
        "Oil": oil,
        "latent_stress": stress,
        "stress_score": stress_score,
        "true_regime": true_regime,
    })

    return df


if __name__ == "__main__":
    df = generate_synthetic_regime_data()
    print(df.head())
    print(df["true_regime"].value_counts())


# ======================================================================
# Quick test when run directly
# ======================================================================
if __name__ == "__main__":
    df = generate_synthetic_regime_data()

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nRegime counts:")
    print(df["true_regime"].value_counts())

    # Regime duration stats
    runs = []
    current = df["true_regime"].iloc[0]
    length = 1
    for t in range(1, len(df)):
        if df["true_regime"].iloc[t] == current:
            length += 1
        else:
            runs.append((current, length))
            current = df["true_regime"].iloc[t]
            length = 1
    runs.append((current, length))

    run_df = pd.DataFrame(runs, columns=["regime", "duration"])
    print(f"\nRegime episodes: {len(run_df)}")
    print(f"Mean duration: {run_df['duration'].mean():.1f} days")
    for r in ["bull", "bear"]:
        sub = run_df[run_df["regime"] == r]
        print(
            f"  {r}: {len(sub)} episodes, "
            f"mean {sub['duration'].mean():.1f} days, "
            f"median {sub['duration'].median():.0f} days"
        )

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    regime_change = (df["true_regime"] != df["true_regime"].shift(1)).cumsum()

    # 1. Price with regime shading
    ax1.plot(df["Date"], df["GSPC"], color="black", linewidth=0.8)
    ax1.set_title("Synthetic GSPC with Ground Truth Regime Shading", fontsize=14)
    ax1.set_ylabel("Price", fontsize=12)

    for _, group in df.groupby(regime_change):
        s = group["Date"].iloc[0]
        e = group["Date"].iloc[-1]
        r = group["true_regime"].iloc[0]
        ax1.axvspan(s, e, color="green" if r == "bull" else "red", alpha=0.2)

    # 2. VIX with regime shading
    ax2.plot(df["Date"], df["VIX"], color="purple", linewidth=0.8)
    ax2.set_title("Synthetic VIX with Ground Truth Regime Shading", fontsize=14)
    ax2.set_ylabel("VIX", fontsize=12)

    for _, group in df.groupby(regime_change):
        s = group["Date"].iloc[0]
        e = group["Date"].iloc[-1]
        r = group["true_regime"].iloc[0]
        ax2.axvspan(s, e, color="green" if r == "bull" else "red", alpha=0.2)

    # 3. Stress score with regime shading
    ax3.plot(df["Date"], df["stress_score"], color="blue", linewidth=0.8)
    ax3.set_title("Composite Stress Score with Ground Truth Regime Shading", fontsize=14)
    ax3.set_ylabel("Stress Score", fontsize=12)

    for _, group in df.groupby(regime_change):
        s = group["Date"].iloc[0]
        e = group["Date"].iloc[-1]
        r = group["true_regime"].iloc[0]
        ax3.axvspan(s, e, color="green" if r == "bull" else "red", alpha=0.2)

    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.set_xlabel("Date", fontsize=12)

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("reports/synthetic/synthetic_regimes.png", dpi=300)
    plt.show()
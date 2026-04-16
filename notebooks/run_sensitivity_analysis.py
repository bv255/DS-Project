"""
Sensitivity Analysis for Regime Detection Validation
======================================================
Runs HMM, GMM, and Changepoint detectors on synthetic data generated
under varying conditions, and collects accuracy metrics into summary tables.

Methodology: Bucci & Ciciretti (2022) first-pillar synthetic validation,
extended with systematic parameter variation.

Requirements:
    pip install hmmlearn scikit-learn ruptures pandas numpy joblib
    synthetic_regime_generator.py must be in the same directory.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hmmlearn import hmm
import ruptures as rpt
from sklearn.metrics import classification_report
from joblib import Parallel, delayed

from .synthetic_regime_data_generator import generate_synthetic_regime_data


REPORT_METRICS = ['accuracy', 'macro_f1', 'bear_f1', 'bull_f1']


# =====================================================================
# DETECTOR FUNCTIONS
# =====================================================================

def run_hmm(df):
    """Run HMM regime detection — matches notebook pipeline."""
    features = ['Return', 'VIX', 'Drawdown', 'RSI_14',
                'MACD_Hist', 'Consumer_Sentiment_ZScore', 'Oil']
    df_clean = df.dropna(subset=features).copy().reset_index(drop=True)
    X_raw = df_clean[features].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = hmm.GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=200, tol=1e-4, random_state=42)
    model.fit(X)
    smoothed_probs = model.predict_proba(X)

    means_original = scaler.inverse_transform(model.means_)
    # Return is the first feature
    if means_original[0][0] > means_original[1][0]:
        bull_state = 0
    else:
        bull_state = 1

    # 120-day rolling smoothing on posterior (matches notebook)
    df_clean['prob_bull'] = smoothed_probs[:, bull_state]
    df_clean['prob_bull_smooth'] = df_clean['prob_bull'].rolling(
        120, min_periods=1, center=True).mean()
    df_clean['regime'] = np.where(df_clean['prob_bull_smooth'] > 0.5, 'bull', 'bear')
    df_clean['true_regime'] = df['true_regime'].values[:len(df_clean)]

    return df_clean[['regime', 'true_regime']].dropna()


def run_gmm(df):
    """Run GMM regime detection — matches notebook pipeline."""
    candidate_features = ['Return', 'VIX', 'Drawdown', 'RSI_14',
                'MACD_Hist', 'Consumer_Sentiment_ZScore', 'Oil']

    df_work = df[['Date', 'Return', 'VIX', 'Drawdown', 'RSI_14',
                  'MACD_Hist', 'Consumer_Sentiment_ZScore', 'Oil']].copy()
    df_work['true_regime'] = df['true_regime'].values[:len(df_work)]

    df_work[candidate_features] = df_work[candidate_features].ffill(limit=5)
    df_work[candidate_features] = df_work[candidate_features].interpolate(method='linear')
    df_work = df_work.dropna(subset=candidate_features).reset_index(drop=True)

    # Collinearity removal
    corr_matrix = df_work[candidate_features].corr()
    upper_tri = corr_matrix.where(
        np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high_corr_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            val = upper_tri.loc[idx, col]
            if pd.notna(val) and abs(val) > 0.8:
                high_corr_pairs.append((idx, col, val))

    features_to_drop = set()
    for f1, f2, r in high_corr_pairs:
        if f1 not in features_to_drop and f2 not in features_to_drop:
            count_f1 = sum(1 for a, b, _ in high_corr_pairs if a == f1 or b == f1)
            count_f2 = sum(1 for a, b, _ in high_corr_pairs if a == f2 or b == f2)
            features_to_drop.add(f1 if count_f1 >= count_f2 else f2)

    selected_features = [f for f in candidate_features if f not in features_to_drop]

    scaler = StandardScaler()
    X = scaler.fit_transform(df_work[selected_features].values)

    gmm = GaussianMixture(n_components=2, covariance_type='full',
                          max_iter=200, n_init=5, random_state=42, tol=1e-4)
    gmm.fit(X)
    proba = gmm.predict_proba(X)
    labels = gmm.predict(X)

    ret_0 = df_work.loc[labels == 0, 'Return'].mean()
    ret_1 = df_work.loc[labels == 1, 'Return'].mean()
    bull_cluster = 0 if ret_0 > ret_1 else 1

    # 120-day rolling smoothing on posterior (matches notebook)
    df_work['prob_bull'] = proba[:, bull_cluster]
    df_work['prob_bull_smooth'] = df_work['prob_bull'].rolling(
        120, min_periods=1, center=True).mean()
    df_work['regime'] = np.where(df_work['prob_bull_smooth'] > 0.5, 'bull', 'bear')

    return df_work[['regime', 'true_regime']].dropna()


def run_changepoint(df):
    """Run PELT changepoint regime detection. Returns predicted regime labels."""
    candidate_features = ['Return', 'VIX', 'Drawdown', 'RSI_14',
                          'MACD_Hist', 'Consumer_Sentiment_ZScore',
                        'Oil']

    df_work = df[['Date', 'GSPC', 'Return', 'VIX', 'Drawdown', 'RSI_14',
                  'MACD_Hist', 'Consumer_Sentiment_ZScore',  'Oil']].copy()
    df_work['true_regime'] = df['true_regime'].values[:len(df_work)]

    # Clean
    df_work[candidate_features] = df_work[candidate_features].ffill(limit=5)
    df_work[candidate_features] = df_work[candidate_features].interpolate(method='linear')
    df_work = df_work.dropna(subset=candidate_features).reset_index(drop=True)

    # Collinearity removal
    corr_matrix = df_work[candidate_features].corr()
    upper_tri = corr_matrix.where(
        np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high_corr_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            val = upper_tri.loc[idx, col]
            if pd.notna(val) and abs(val) > 0.8:
                high_corr_pairs.append((idx, col, val))

    features_to_drop = set()
    for f1, f2, r in high_corr_pairs:
        if f1 not in features_to_drop and f2 not in features_to_drop:
            count_f1 = sum(1 for a, b, _ in high_corr_pairs if a == f1 or b == f1)
            count_f2 = sum(1 for a, b, _ in high_corr_pairs if a == f2 or b == f2)
            features_to_drop.add(f1 if count_f1 >= count_f2 else f2)

    selected_features = [f for f in candidate_features if f not in features_to_drop]

    scaler = StandardScaler()
    X = scaler.fit_transform(df_work[selected_features].values)

    n = len(X)
    k = X.shape[1]

    # PELT with BIC-based penalty selection and minimum segment length
    algo = rpt.Pelt(model="rbf", min_size=20).fit(X)

    # Pre-compute all candidate breakpoint sets, then score
    candidates = {}
    for pen in [1, 3, 5, 10, 20, 35, 50]:
        candidates[pen] = algo.predict(pen=pen)

    best_pen = 5  # fallback
    best_bic = np.inf
    for pen, bkps in candidates.items():
        n_segments = len(bkps)
        cost = algo.cost.sum_of_costs(bkps)
        bic = cost + n_segments * k * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_pen = pen

    result = candidates[best_pen]

    # Assign regimes by mean return per segment
    df_work['regime'] = 'unknown'
    start_idx = 0
    for end_idx in result:
        end_idx = min(end_idx, len(df_work))
        segment_data = df_work.iloc[start_idx:end_idx]
        mean_ret = segment_data['Return'].mean()
        regime = 'bull' if mean_ret > 0 else 'bear'
        df_work.loc[start_idx:end_idx - 1, 'regime'] = regime
        start_idx = end_idx

    return df_work[['regime', 'true_regime']].dropna()


# =====================================================================
# EVALUATION FUNCTION
# =====================================================================

def evaluate(result_df):
    predicted = result_df['regime'].values
    true = result_df['true_regime'].values

    mask = pd.notna(predicted) & pd.notna(true)
    predicted = predicted[mask]
    true = true[mask]

    report = classification_report(true, predicted, output_dict=True, zero_division=0)

    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'bear_f1':  report['bear']['f1-score'],
        'bull_f1':  report['bull']['f1-score'],
    }

# =====================================================================
# PARALLELISED SENSITIVITY EXPERIMENTS
# =====================================================================

MODELS = [('HMM', run_hmm), ('GMM', run_gmm), ('Changepoint', run_changepoint)]


def _run_one_seed(val, seed, param_name, generator_kwargs_base):
    kwargs = {**generator_kwargs_base, param_name: val, 'seed': seed}
    df = generate_synthetic_regime_data(**kwargs)

    results = {}
    for model_name, model_fn in MODELS:
        try:
            result = model_fn(df)
            metrics = evaluate(result)
        except Exception:
            metrics = {m: np.nan for m in REPORT_METRICS}
        results[model_name] = metrics

    # Per-model rows
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            param_name: val,
            'seed': seed,
            'model': model_name,
            **metrics,
        })

    # Pairwise comparison rows
    comp_rows = []
    pairs = [
        ('Changepoint', 'GMM', 'PELT vs GMM'),
        ('Changepoint', 'HMM', 'PELT vs HMM'),
    ]
    for model_a, model_b, pair_name in pairs:
        ma = results.get(model_a)
        mb = results.get(model_b)
        if ma is None or mb is None:
            continue
        row = {param_name: val, 'seed': seed, 'comparison': pair_name}
        for metric in REPORT_METRICS:
            row[f'{metric}_diff'] = ma[metric] - mb[metric]
        comp_rows.append(row)

    return rows, comp_rows

def run_baseline(n_seeds=10):
    """Run all detectors at default parameters across multiple seeds."""
    all_rows = []
    all_comp = []

    for seed in range(n_seeds):
        df = generate_synthetic_regime_data(seed=seed)

        results = {}
        for model_name, model_fn in MODELS:
            try:
                result = model_fn(df)
                metrics = evaluate(result)
            except Exception:
                metrics = {m: np.nan for m in REPORT_METRICS}
            results[model_name] = metrics

        for model_name, metrics in results.items():
            all_rows.append({'seed': seed, 'model': model_name, **metrics})

        for model_a, model_b, pair_name in [
            ('Changepoint', 'GMM', 'PELT vs GMM'),
            ('Changepoint', 'HMM', 'PELT vs HMM'),
        ]:
            row = {'seed': seed, 'comparison': pair_name}
            for metric in REPORT_METRICS:
                row[f'{metric}_diff'] = results[model_a][metric] - results[model_b][metric]
            all_comp.append(row)

    raw_df = pd.DataFrame(all_rows)
    comp_df = pd.DataFrame(all_comp)

    print(f"\n{'='*90}")
    print(f"Baseline results (mean +/- std over {n_seeds} seeds)")
    print(f"{'='*90}")
    for model_name in ['Changepoint', 'GMM', 'HMM']:
        m = raw_df[raw_df['model'] == model_name][REPORT_METRICS]
        parts = [f"{col}: {m[col].mean():.3f}+/-{m[col].std():.3f}"
                 for col in REPORT_METRICS]
        print(f"  {model_name:12s} | {' | '.join(parts)}")

    print(f"\n  Pairwise (PELT minus comparator):")
    for pair_name in ['PELT vs GMM', 'PELT vs HMM']:
        group = comp_df[comp_df['comparison'] == pair_name]
        parts = []
        for metric in REPORT_METRICS:
            diffs = group[f'{metric}_diff'].values
            parts.append(f"{metric}: {diffs.mean():+.3f}+/-{diffs.std():.3f} "
                         f"({(diffs > 0).mean():.0%})")
        print(f"    {pair_name:14s} | {' | '.join(parts)}")

    return raw_df, comp_df

def aggregate_all_results(baseline_comp, exp1_comp, exp2_comp, exp3_comp):
    def summarise_comp(comp_df, param_col=None):
        rows = []
        values = sorted(comp_df[param_col].unique()) if param_col else ['default']
        for val in values:
            subset = comp_df[comp_df[param_col] == val] if param_col else comp_df
            for pair_name in ['PELT vs GMM', 'PELT vs HMM']:
                group = subset[subset['comparison'] == pair_name]
                if len(group) == 0:
                    continue
                row = {'param_value': val, 'comparison': pair_name}
                for metric in REPORT_METRICS:
                    diffs = group[f'{metric}_diff'].values
                    row[f'{metric}_diff_mean'] = diffs.mean()
                    row[f'{metric}_diff_std'] = diffs.std()
                    row[f'{metric}_wins'] = (diffs > 0).mean()
                rows.append(row)
        return rows

    all_rows = []
    for r in summarise_comp(baseline_comp):
        r['experiment'] = 'Baseline'
        all_rows.append(r)
    for comp_df, param_col, name in [
        (exp1_comp, 'bear_std_return', 'Volatility Gap'),
        (exp2_comp, 'ar_coeff', 'Regime Persistence'),
        (exp3_comp, 'bear_vix_mean', 'VIX Signal'),
    ]:
        for r in summarise_comp(comp_df, param_col):
            r['experiment'] = name
            all_rows.append(r)

    return pd.DataFrame(all_rows)


def run_experiment(param_name, param_values, generator_kwargs_base=None,
                   n_seeds=10, n_jobs=-1):
    if generator_kwargs_base is None:
        generator_kwargs_base = {}

    print(f"\nDispatching {len(param_values)} x {n_seeds} = "
          f"{len(param_values) * n_seeds} jobs ...")

    nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_one_seed)(val, seed, param_name, generator_kwargs_base)
        for val in param_values
        for seed in range(n_seeds)
    )

    raw_rows = [r for batch_rows, _ in nested for r in batch_rows]
    comp_rows = [r for _, batch_comp in nested for r in batch_comp]

    raw_df = pd.DataFrame(raw_rows)
    comp_df = pd.DataFrame(comp_rows)

    # --- Per-model summary ---
    print(f"\n{'='*90}")
    print(f"Per-model results (mean +/- std over {n_seeds} seeds)")
    print(f"{'='*90}")
    for val in param_values:
        print(f"\n  {param_name} = {val}")
        subset = raw_df[raw_df[param_name] == val]
        for model_name in ['Changepoint', 'GMM', 'HMM']:
            m = subset[subset['model'] == model_name][REPORT_METRICS]
            if len(m) == 0:
                continue
            parts = [f"{col}: {m[col].mean():.3f}+/-{m[col].std():.3f}"
                     for col in REPORT_METRICS]
            print(f"    {model_name:12s} | {' | '.join(parts)}")

    # --- Pairwise comparison summary ---
    print(f"\n{'='*90}")
    print(f"Pairwise comparisons (PELT minus comparator, mean diff +/- std, win rate)")
    print(f"{'='*90}")

    comp_summary_rows = []
    for val in param_values:
        print(f"\n  {param_name} = {val}")
        val_comp = comp_df[comp_df[param_name] == val]
        for pair_name in ['PELT vs GMM', 'PELT vs HMM']:
            group = val_comp[val_comp['comparison'] == pair_name]
            if len(group) == 0:
                continue
            row = {param_name: val, 'comparison': pair_name}
            parts = []
            for metric in REPORT_METRICS:
                diffs = group[f'{metric}_diff'].values
                row[f'{metric}_diff_mean'] = diffs.mean()
                row[f'{metric}_diff_std'] = diffs.std()
                row[f'{metric}_pelt_wins'] = (diffs > 0).mean()
                parts.append(f"{metric}: {diffs.mean():+.3f}+/-{diffs.std():.3f} "
                             f"({(diffs > 0).mean():.0%})")
            comp_summary_rows.append(row)
            print(f"    {pair_name:14s} | {' | '.join(parts)}")

    comp_summary = pd.DataFrame(comp_summary_rows)

    return raw_df, comp_df, comp_summary

# =====================================================================
# MAIN — Run all three experiments
# =====================================================================

if __name__ == '__main__':

    N_SEEDS = 10

    # --- Baseline ---
    print("\n" + "#" * 70)
    print("# BASELINE: DEFAULT PARAMETERS")
    print(f"# {N_SEEDS} seeds")
    print("#" * 70)

    baseline_raw, baseline_comp = run_baseline(n_seeds=N_SEEDS)
    baseline_raw.to_csv('sensitivity_baseline_raw.csv', index=False)
    baseline_comp.to_csv('sensitivity_baseline_comparisons.csv', index=False)

    # --- Experiment 1: Volatility Gap ---
    print("\n" + "#" * 70)
    print("# EXPERIMENT 1: VOLATILITY GAP")
    print("# Varying bear_std_return while bull_std_return stays at 0.008")
    print(f"# {N_SEEDS} seeds per setting")
    print("#" * 70)

    exp1_raw, exp1_comp, exp1_summary = run_experiment(
        param_name='bear_std_return',
        param_values=[0.010, 0.014, 0.018, 0.022, 0.030],
        n_seeds=N_SEEDS,
    )

    exp2_raw, exp2_comp, exp2_summary = run_experiment(
        param_name='ar_coeff',
        param_values=[0.5, 0.7, 0.9, 0.95, 0.99],
        n_seeds=N_SEEDS,
    )

    exp3_raw, exp3_comp, exp3_summary = run_experiment(
        param_name='bear_vix_mean',
        param_values=[17.0, 20.0, 25.0, 30.0, 40.0],
        n_seeds=N_SEEDS,
    )

    # --- Save all results ---
    exp1_summary.to_csv('sensitivity_exp1_volatility_gap.csv', index=False)
    exp2_summary.to_csv('sensitivity_exp2_regime_persistence.csv', index=False)
    exp3_summary.to_csv('sensitivity_exp3_vix_signal.csv', index=False)

    exp1_raw.to_csv('sensitivity_exp1_volatility_gap_raw.csv', index=False)
    exp2_raw.to_csv('sensitivity_exp2_regime_persistence_raw.csv', index=False)
    exp3_raw.to_csv('sensitivity_exp3_vix_signal_raw.csv', index=False)

    exp1_comp.to_csv('sensitivity_exp1_volatility_gap_comparisons.csv', index=False)
    exp2_comp.to_csv('sensitivity_exp2_regime_persistence_comparisons.csv', index=False)
    exp3_comp.to_csv('sensitivity_exp3_vix_signal_comparisons.csv', index=False)

    # --- Aggregate summary ---
    summary = aggregate_all_results(baseline_comp, exp1_comp, exp2_comp, exp3_comp)
    summary.to_csv('sensitivity_full_summary.csv', index=False)

    print("\n" + "=" * 90)
    print("SUMMARY: PELT win rate on Macro F1 across all conditions")
    print("=" * 90)

    for exp_name in summary['experiment'].unique():
        print(f"\n  {exp_name}:")
        exp_sub = summary[summary['experiment'] == exp_name]
        for _, r in exp_sub.iterrows():
            wins = r['macro_f1_wins']
            marker = '<--' if wins < 0.3 else '***' if wins > 0.7 else ''
            print(f"    {r['comparison']:14s} | param={str(r['param_value']):<8s} | "
                  f"macro_f1 diff: {r['macro_f1_diff_mean']:+.3f} | "
                  f"win rate: {wins:.0%} {marker}")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"({N_SEEDS} seeds per parameter setting)")
    print("=" * 70)
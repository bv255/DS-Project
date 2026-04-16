"""
Sensitivity Analysis Results Plotter
=====================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'HMM': '#2ecc71', 'GMM': '#3498db', 'Changepoint': '#e74c3c'}
MARKERS = {'HMM': 'o', 'GMM': 's', 'Changepoint': '^'}

REPORT_METRICS = ['accuracy', 'macro_f1', 'bear_f1', 'bull_f1']


def plot_metric_comparison(df, param_col, param_label, metric='accuracy',
                           title=None, ax=None, show_std=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    for model in ['HMM', 'GMM', 'Changepoint']:
        subset = df[df['model'] == model].sort_values(param_col)
        x = subset[param_col].values
        y = subset[f'{metric}_mean'].values
        
        ax.plot(x, y, marker=MARKERS[model], color=COLORS[model],
                label=model, linewidth=2, markersize=8)
        
        if show_std and f'{metric}_std' in subset.columns:
            std = subset[f'{metric}_std'].values
            ax.fill_between(x, y - std, y + std, color=COLORS[model], alpha=0.15)
    
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(title or f'{metric.title()} vs {param_label}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.set_ylim(0, 1.05)
    return ax


def plot_experiment_summary(df, param_col, param_label, exp_title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(exp_title, fontsize=14, fontweight='bold', y=1.02)
    
    metrics = [
        ('accuracy', 'Overall Accuracy'),
        ('macro_f1', 'Macro F1 Score'),
        ('bear_f1', 'Bear Regime F1'),
        ('bull_f1', 'Bull Regime F1'),
    ]
    
    for ax, (metric, title) in zip(axes.flat, metrics):
        plot_metric_comparison(df, param_col, param_label, metric=metric,
                               title=title, ax=ax)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_all_experiments_single_metric(exp1, exp2, exp3, metric='macro_f1',
                                        filename='sensitivity_comparison.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    configs = [
        (exp1, 'bear_std_return', 'Bear Volatility (σ)', 'Exp 1: Volatility Gap'),
        (exp2, 'ar_coeff', 'AR Coefficient', 'Exp 2: Regime Persistence'),
        (exp3, 'bear_vix_mean', 'Bear VIX Mean', 'Exp 3: VIX Signal Strength'),
    ]
    
    for ax, (df, param_col, param_label, title) in zip(axes, configs):
        plot_metric_comparison(df, param_col, param_label, metric=metric,
                               title=title, ax=ax)
    
    fig.suptitle(f'{metric.replace("_", " ").title()} Across Sensitivity Experiments',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_heatmap(df, param_col, metric='macro_f1', filename='heatmap.png'):
    pivot = df.pivot(index='model', columns=param_col, values=f'{metric}_mean')
    pivot = pivot.reindex(['HMM', 'GMM', 'Changepoint'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.3g}' for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label=metric.replace('_', ' ').title())
    ax.set_xlabel(param_col.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Heatmap', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def load_and_aggregate(raw_csv_path, param_col):
    raw = pd.read_csv(raw_csv_path)
    rows = []
    for (val, model), group in raw.groupby([param_col, 'model']):
        row = {param_col: val, 'model': model}
        for m in REPORT_METRICS:
            row[f'{m}_mean'] = group[m].mean()
            row[f'{m}_std'] = group[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def load_comparisons(csv_path, param_col=None):
    comp = pd.read_csv(csv_path)
    
    if param_col:
        values = sorted(comp[param_col].unique())
    else:
        values = ['default']
    
    rows = []
    for val in values:
        subset = comp[comp[param_col] == val] if param_col else comp
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
    return pd.DataFrame(rows)


def main():
    output_dir = 'reports/synthetic'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # --- Load and aggregate per-model results ---
    try:
        exp1 = load_and_aggregate('sensitivity_exp1_volatility_gap_raw.csv', 'bear_std_return')
        exp2 = load_and_aggregate('sensitivity_exp2_regime_persistence_raw.csv', 'ar_coeff')
        exp3 = load_and_aggregate('sensitivity_exp3_vix_signal_raw.csv', 'bear_vix_mean')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run sensitivity_analysis.py first.")
        return
    
    # --- Plots ---
    plot_experiment_summary(
        exp1, 'bear_std_return', 'Bear Volatility (σ)',
        'Experiment 1: Volatility Gap Sensitivity',
        f'{output_dir}/exp1_volatility_gap.png'
    )
    plot_experiment_summary(
        exp2, 'ar_coeff', 'AR Coefficient',
        'Experiment 2: Regime Persistence Sensitivity',
        f'{output_dir}/exp2_regime_persistence.png'
    )
    plot_experiment_summary(
        exp3, 'bear_vix_mean', 'Bear VIX Mean',
        'Experiment 3: VIX Signal Strength Sensitivity',
        f'{output_dir}/exp3_vix_signal.png'
    )
    
    plot_all_experiments_single_metric(exp1, exp2, exp3, metric='macro_f1',
                                        filename=f'{output_dir}/sensitivity_macro_f1_comparison.png')
    plot_all_experiments_single_metric(exp1, exp2, exp3, metric='accuracy',
                                        filename=f'{output_dir}/sensitivity_accuracy_comparison.png')
    
    plot_heatmap(exp1, 'bear_std_return', metric='macro_f1',
                 filename=f'{output_dir}/heatmap_exp1_volatility.png')
    plot_heatmap(exp2, 'ar_coeff', metric='macro_f1',
                 filename=f'{output_dir}/heatmap_exp2_persistence.png')
    plot_heatmap(exp3, 'bear_vix_mean', metric='macro_f1',
                 filename=f'{output_dir}/heatmap_exp3_vix.png')
    
    # --- Pairwise comparison summary ---
    try:
        bl_comp = load_comparisons('sensitivity_baseline_comparisons.csv')
        e1_comp = load_comparisons('sensitivity_exp1_volatility_gap_comparisons.csv', 'bear_std_return')
        e2_comp = load_comparisons('sensitivity_exp2_regime_persistence_comparisons.csv', 'ar_coeff')
        e3_comp = load_comparisons('sensitivity_exp3_vix_signal_comparisons.csv', 'bear_vix_mean')
    except FileNotFoundError as e:
        print(f"\nSkipping pairwise summary: {e}")
        print("All plots saved.")
        return
    
    # Tag each with experiment name and concatenate
    bl_comp['experiment'] = 'Baseline'
    e1_comp['experiment'] = 'Volatility Gap'
    e2_comp['experiment'] = 'Regime Persistence'
    e3_comp['experiment'] = 'VIX Signal'
    
    summary = pd.concat([bl_comp, e1_comp, e2_comp, e3_comp], ignore_index=True)
    summary.to_csv(f'{output_dir}/sensitivity_full_summary.csv', index=False)
    
    # Print
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
    
    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == '__main__':
    main()
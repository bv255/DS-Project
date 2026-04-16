import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Load the datasets
pelt_df = pd.read_csv('.data/multivariate_changepoint_labeled_dataset.csv')
gmm_df  = pd.read_csv('data/gmm_labeled_dataset.csv')
hmm_df  = pd.read_csv('.data/hmm_labeled_dataset.csv')

# Load NBER ground truth — adjust path/column names as needed
nber_df = pd.read_csv('data/nber_ground_truth.csv')

# Align on date
pelt_df['date'] = pd.to_datetime(pelt_df['date'])
gmm_df['date']  = pd.to_datetime(gmm_df['date'])
hmm_df['date']  = pd.to_datetime(hmm_df['date'])
nber_df['date']  = pd.to_datetime(nber_df['date'])

# Merge everything on date
merged = pelt_df[['date', 'regime']].rename(columns={'regime': 'pred_pelt'})
merged = merged.merge(gmm_df[['date', 'regime']].rename(columns={'regime': 'pred_gmm'}), on='date')
merged = merged.merge(hmm_df[['date', 'regime']].rename(columns={'regime': 'pred_hmm'}), on='date')
merged = merged.merge(nber_df[['date', 'label']].rename(columns={'label': 'y_true'}), on='date')

print(f"Merged dataset: {len(merged)} days")

# Correctness arrays
correct_pelt = (merged['pred_pelt'] == merged['y_true'])
correct_gmm  = (merged['pred_gmm']  == merged['y_true'])
correct_hmm  = (merged['pred_hmm']  == merged['y_true'])

# McNemar's test
def build_table(ca, cb):
    n11 = np.sum( ca &  cb)
    n10 = np.sum( ca & ~cb)
    n01 = np.sum(~ca &  cb)
    n00 = np.sum(~ca & ~cb)
    return np.array([[n11, n10],
                     [n01, n00]])

pairs = [
    ("PELT vs GMM", correct_pelt, correct_gmm),
    ("PELT vs HMM", correct_pelt, correct_hmm),
    ("GMM vs HMM",  correct_gmm,  correct_hmm),
]

print("\n=== McNemar's Test (vs NBER) ===\n")
for name, ca, cb in pairs:
    table = build_table(ca, cb)
    result = mcnemar(table, exact=False, correction=True)
    adjusted_p = min(result.pvalue * 3, 1.0)
    
    print(f"{name}")
    print(f"  Off-diagonal: b={table[0,1]}, c={table[1,0]}")
    print(f"  Chi-squared:  {result.statistic:.2f}")
    print(f"  p-value:      {result.pvalue:.2e}")
    print(f"  Bonferroni p: {adjusted_p:.2e}")
    sig = "YES" if adjusted_p < 0.05 else "NO"
    print(f"  Significant at 0.05: {sig}\n")
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import EXPERIMENTS_DIR, RESULTS_DIR

# Load all results
with open(EXPERIMENTS_DIR / "baseline_zero_shot_reward.json") as f:
    zero_shot = json.load(f)
with open(EXPERIMENTS_DIR / "baseline_few_shot_reward.json") as f:
    few_shot = json.load(f)
with open(EXPERIMENTS_DIR / "single_domain_crpo_reward.json") as f:
    single_crpo = json.load(f)
with open(EXPERIMENTS_DIR / "multi_domain_crpo_reward.json") as f:
    multi_crpo = json.load(f)

# Map baseline domain names to CRPO domain names
domain_mapping = {
    'gsm8k': 'math',
    'bbh_navigate': 'reasoning',
    'bbh_boolean': 'reasoning',
    'liar': 'fact',
    'code': 'code'
}

# Create results table using only the first occurrence of each domain
results_table = []
processed_domains = set()

for baseline_domain, crpo_domain in domain_mapping.items():
    if crpo_domain in processed_domains:
        continue  # Skip duplicates (we already processed this CRPO domain)
    
    processed_domains.add(crpo_domain)
    
    row = {
        'Domain': baseline_domain.replace('_', ' ').title(),
        'Zero-Shot': zero_shot[baseline_domain]['average_score'],
        'Few-Shot': few_shot[baseline_domain]['average_score'],
        'Single-Domain CRPO': single_crpo[crpo_domain]['evaluation']['average_score'],
        'Multi-Domain CRPO': multi_crpo['evaluations'][crpo_domain]['average_score']
    }
    results_table.append(row)

df_results = pd.DataFrame(results_table)

# Print results
print("\n" + "=" * 120)
print("COMPREHENSIVE RESULTS ANALYSIS (REWARD MODEL SCORES)")
print("=" * 120)

print("\nPER-DOMAIN SCORES:")
print(df_results.to_string(index=False))

# Calculate statistics
print("\n\nAVERAGE SCORES BY METHOD:")
avg_zero = df_results['Zero-Shot'].mean()
avg_few = df_results['Few-Shot'].mean()
avg_single = df_results['Single-Domain CRPO'].mean()
avg_multi = df_results['Multi-Domain CRPO'].mean()

print(f"  Zero-Shot: {avg_zero:.3f}")
print(f"  Few-Shot: {avg_few:.3f}")
print(f"  Single-Domain CRPO: {avg_single:.3f}")
print(f"  Multi-Domain CRPO: {avg_multi:.3f}")

# Robustness (standard deviation)
print("\n\nROBUSTNESS SCORES (Lower is Better - std dev across domains):")
robustness_zero = df_results['Zero-Shot'].std()
robustness_few = df_results['Few-Shot'].std()
robustness_single = df_results['Single-Domain CRPO'].std()
robustness_multi = df_results['Multi-Domain CRPO'].std()

print(f"  Zero-Shot: {robustness_zero:.4f}")
print(f"  Few-Shot: {robustness_few:.4f}")
print(f"  Single-Domain CRPO: {robustness_single:.4f}")
print(f"  Multi-Domain CRPO: {robustness_multi:.4f}")

# Key finding
improvement = (1 - robustness_multi / robustness_single) * 100 if robustness_single > 0 else 0
print(f"\n  ROBUSTNESS IMPROVEMENT: {improvement:.1f}%")

# Save tables
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df_results.to_csv(RESULTS_DIR / "final_results_table_reward.csv", index=False)

analysis_data = {
    'per_domain': df_results.to_dict('records'),
    'averages': {
        'Zero-Shot': float(avg_zero),
        'Few-Shot': float(avg_few),
        'Single-Domain CRPO': float(avg_single),
        'Multi-Domain CRPO': float(avg_multi)
    },
    'robustness': {
        'Zero-Shot': float(robustness_zero),
        'Few-Shot': float(robustness_few),
        'Single-Domain CRPO': float(robustness_single),
        'Multi-Domain CRPO': float(robustness_multi)
    },
    'robustness_improvement_percent': float(improvement)
}

with open(RESULTS_DIR / "final_analysis_reward.json", "w") as f:
    json.dump(analysis_data, f, indent=2)

print("\n" + "=" * 60)
print(" Results saved to:")
print(f"  - {RESULTS_DIR}/final_results_table_reward.csv")
print(f"  - {RESULTS_DIR}/final_analysis_reward.json")

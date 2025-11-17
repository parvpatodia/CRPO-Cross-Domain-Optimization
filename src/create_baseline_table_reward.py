import json
import pandas as pd
import numpy as np
import os

# Load all baseline results - simple paths since results/ is now in src/
with open("experiments/baseline_zero_shot_reward.json") as f:
    zero_shot = json.load(f)

with open("experiments/baseline_few_shot_reward.json") as f:
    few_shot = json.load(f)

with open("experiments/single_domain_crpo_reward.json") as f:
    single_crpo = json.load(f)

print("✓ Loaded all experiment files")

# Map domain names
domain_mapping = {
    'gsm8k': 'math',
    'bbh_navigate': 'reasoning',
    'bbh_boolean': 'reasoning',
    'liar': 'fact',
    'code': 'code'
}

data = []

for baseline_domain in ['gsm8k', 'bbh_navigate', 'bbh_boolean', 'liar', 'code']:
    crpo_domain = domain_mapping[baseline_domain]
    
    row = {
        'Domain': baseline_domain.replace('_', ' ').title(),
        'Zero-Shot': f"{zero_shot[baseline_domain]['average_score']:.3f}",
        'Few-Shot': f"{few_shot[baseline_domain]['average_score']:.3f}",
        'Single-Domain CRPO': f"{single_crpo[crpo_domain]['evaluation']['average_score']:.3f}"
    }
    data.append(row)

df = pd.DataFrame(data)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Save table
df.to_csv("results/baseline_comparison_reward.csv", index=False)

# Print table
print("\n" + "=" * 100)
print("BASELINE COMPARISON TABLE (Using Reward Model)")
print("=" * 100)
print(df.to_string(index=False))
print("=" * 100)

# Save as JSON
with open("results/baseline_comparison_reward.json", "w") as f:
    json.dump(df.to_dict('records'), f, indent=2)

print("\nTable saved to results/")

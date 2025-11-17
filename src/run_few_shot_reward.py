import json
import os
import numpy as np
from dotenv import load_dotenv
from evaluation_reward import EvaluatorWithRewardModel
from data_loader import DataLoader
from few_shot_examples import FEW_SHOT_EXAMPLES

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

evaluator = EvaluatorWithRewardModel(api_key)
loader = DataLoader()

print("=" * 60)
print("FEW-SHOT MANUAL BASELINE (With Reward Model)")
print("=" * 60)

results = {}

# 1. GSM8K
print("\n1. GSM8K Test Set")
gsm8k_test = loader.load_gsm8k('test')
gsm8k_result = evaluator.evaluate_dataset(
    FEW_SHOT_EXAMPLES['math'],
    gsm8k_test,
    'math',
    max_examples=100
)
results['gsm8k'] = gsm8k_result
print(f"   Average Score: {gsm8k_result['average_score']:.3f}")
print(f"   Std Dev: {gsm8k_result['std_dev']:.3f}")

# 2. BBH Navigate
print("\n2. BBH Navigate Test Set")
_, bbh_nav_test = loader.load_bbh('navigate')
bbh_nav_result = evaluator.evaluate_dataset(
    FEW_SHOT_EXAMPLES['reasoning'],
    bbh_nav_test,
    'reasoning',
    max_examples=50
)
results['bbh_navigate'] = bbh_nav_result
print(f"   Average Score: {bbh_nav_result['average_score']:.3f}")
print(f"   Std Dev: {bbh_nav_result['std_dev']:.3f}")

# 3. BBH Boolean
print("\n3. BBH Boolean Test Set")
_, bbh_bool_test = loader.load_bbh('boolean_expressions')
bbh_bool_result = evaluator.evaluate_dataset(
    FEW_SHOT_EXAMPLES['reasoning'],
    bbh_bool_test,
    'reasoning',
    max_examples=50
)
results['bbh_boolean'] = bbh_bool_result
print(f"   Average Score: {bbh_bool_result['average_score']:.3f}")
print(f"   Std Dev: {bbh_bool_result['std_dev']:.3f}")

# 4. LIAR
print("\n4. LIAR Test Set")
liar_test = loader.load_liar('test')
liar_result = evaluator.evaluate_dataset(
    FEW_SHOT_EXAMPLES['fact_verification'],
    liar_test,
    'fact_verification',
    max_examples=50
)
results['liar'] = liar_result
print(f"   Average Score: {liar_result['average_score']:.3f}")
print(f"   Std Dev: {liar_result['std_dev']:.3f}")

# 5. Code
print("\n5. HumanEval Test Set")
code_data = loader.load_humaneval(50)
code_result = evaluator.evaluate_dataset(
    FEW_SHOT_EXAMPLES['code'],
    code_data[25:],
    'code',
    max_examples=25
)
results['code'] = code_result
print(f"   Average Score: {code_result['average_score']:.3f}")
print(f"   Std Dev: {code_result['std_dev']:.3f}")

# Save results
with open("experiments/baseline_few_shot_reward.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("FEW-SHOT BASELINE COMPLETE")
print(f"Total API calls used: {evaluator.api_calls}")
print(f"Results saved to: experiments/baseline_few_shot_reward.json")

# Print summary
print("\nSUMMARY:")
for domain, result in results.items():
    print(f"{domain:15} | Score: {result['average_score']:.3f} ± {result['std_dev']:.3f}")
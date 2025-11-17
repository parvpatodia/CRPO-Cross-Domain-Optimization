import json
import os
from dotenv import load_dotenv
from evaluation import Evaluator
from data_loader import DataLoader

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

evaluator = Evaluator(api_key)
loader = DataLoader()

# Zero-shot prompt
zero_shot_prompt = "Let's think step by step.\n\nQuestion: {question}\n\nAnswer:"

print("=" * 60)
print("ZERO-SHOT CHAIN-OF-THOUGHT BASELINE")
print("=" * 60)

results = {}
api_call_budget = 0

# 1. GSM8K
print("\n1. GSM8K Test Set")
gsm8k_test = loader.load_gsm8k('test')
gsm8k_result = evaluator.evaluate_dataset(
    zero_shot_prompt, 
    gsm8k_test, 
    'math',
    max_examples=200  # Limit for budget
)
results['gsm8k'] = gsm8k_result
print(f"   Accuracy: {gsm8k_result['accuracy']:.2%}")
print(f"   API calls: {gsm8k_result['api_calls_used']}")

# 2. BBH Navigate
print("\n2. BBH Navigate Test Set")
_, bbh_nav_test = loader.load_bbh('navigate')
bbh_nav_result = evaluator.evaluate_dataset(
    zero_shot_prompt,
    bbh_nav_test,
    'reasoning',
    max_examples=100
)
results['bbh_navigate'] = bbh_nav_result
print(f"   Accuracy: {bbh_nav_result['accuracy']:.2%}")

# 3. BBH Boolean
print("\n3. BBH Boolean Test Set")
_, bbh_bool_test = loader.load_bbh('boolean_expressions')
bbh_bool_result = evaluator.evaluate_dataset(
    zero_shot_prompt,
    bbh_bool_test,
    'reasoning',
    max_examples=100
)
results['bbh_boolean'] = bbh_bool_result
print(f"   Accuracy: {bbh_bool_result['accuracy']:.2%}")

# 4. LIAR
print("\n4. LIAR Test Set")
liar_test = loader.load_liar('test')
liar_result = evaluator.evaluate_dataset(
    zero_shot_prompt,
    liar_test,
    'fact_verification',
    max_examples=100
)
results['liar'] = liar_result
print(f"   Accuracy: {liar_result['accuracy']:.2%}")

# 5. Code
print("\n5. HumanEval Test Set")
code_data = loader.load_humaneval(50)
code_result = evaluator.evaluate_dataset(
    zero_shot_prompt,
    code_data[25:],
    'code',
    max_examples=25
)
results['code'] = code_result
print(f"   Accuracy: {code_result['accuracy']:.2%}")

# Save results
os.makedirs("experiments", exist_ok=True)
with open("experiments/baseline_zero_shot.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("ZERO-SHOT BASELINE COMPLETE")
print(f"Total API calls used: {evaluator.api_calls}")
print(f"Results saved to: experiments/baseline_zero_shot.json")
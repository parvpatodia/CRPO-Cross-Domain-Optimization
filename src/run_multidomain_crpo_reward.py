import json
import os
import numpy as np
from dotenv import load_dotenv
from crpo_multidomain import CRPOMultiDomain
from data_loader import DataLoader
from evaluation_reward import EvaluatorWithRewardModel

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

crpo_multi = CRPOMultiDomain(api_key)
evaluator = EvaluatorWithRewardModel(api_key)
loader = DataLoader()

print("=" * 70)
print("MULTI-DOMAIN CRPO WITH REWARD MODEL")
print("=" * 70)

# Load HelpSteer2
helpsteer2 = loader.load_helpsteer2()
print(f"\nLoaded HelpSteer2 with {len(helpsteer2)} reference examples")

# Define tasks
tasks = {
    'math': 'Solve grade-school math word problems',
    'reasoning': 'Solve logical reasoning tasks',
    'fact': 'Verify statement truthfulness',
    'code': 'Generate correct Python code'
}

# Run multi-domain CRPO
print("\n" + "=" * 70)
print("OPTIMIZING ACROSS ALL DOMAINS")
print("=" * 70)

result_multi = crpo_multi.optimize_multidomain(helpsteer2, tasks)

# Evaluate on all domains
print("\n" + "=" * 70)
print("EVALUATING MULTI-DOMAIN PROMPT ON ALL DOMAINS")
print("=" * 70)

optimized_prompt = result_multi['optimized_prompt']
evaluation_results = {}

# 1. Math
print("\n1. Evaluating on GSM8K (Math)...")
gsm8k_test = loader.load_gsm8k('test')[:100]
eval_math = evaluator.evaluate_dataset(
    optimized_prompt + "\n\nQuestion: {question}\n\nAnswer:",
    gsm8k_test,
    'math',
    max_examples=100
)
evaluation_results['math'] = {
    'average_score': eval_math['average_score'],
    'std_dev': eval_math['std_dev']
}
print(f"   Math score: {eval_math['average_score']:.3f}")

# 2. Reasoning
print("\n2. Evaluating on BBH Navigate (Reasoning)...")
_, bbh_nav_test = loader.load_bbh('navigate')
bbh_nav_test = bbh_nav_test[:50]
eval_reasoning = evaluator.evaluate_dataset(
    optimized_prompt + "\n\nQuestion: {question}\n\nAnswer:",
    bbh_nav_test,
    'reasoning',
    max_examples=50
)
evaluation_results['reasoning'] = {
    'average_score': eval_reasoning['average_score'],
    'std_dev': eval_reasoning['std_dev']
}
print(f"   Reasoning score: {eval_reasoning['average_score']:.3f}")

# 3. Fact
print("\n3. Evaluating on LIAR (Fact Verification)...")
liar_test = loader.load_liar('test')[:50]
eval_fact = evaluator.evaluate_dataset(
    optimized_prompt + "\n\nStatement: {question}\n\nAnswer:",
    liar_test,
    'fact_verification',
    max_examples=50
)
evaluation_results['fact'] = {
    'average_score': eval_fact['average_score'],
    'std_dev': eval_fact['std_dev']
}
print(f"   Fact verification score: {eval_fact['average_score']:.3f}")

# 4. Code
print("\n4. Evaluating on HumanEval (Code)...")
code_data = loader.load_humaneval(50)
code_test = code_data[25:]
eval_code = evaluator.evaluate_dataset(
    optimized_prompt + "\n\nProblem: {question}\n\nSolution:",
    code_test,
    'code',
    max_examples=25
)
evaluation_results['code'] = {
    'average_score': eval_code['average_score'],
    'std_dev': eval_code['std_dev']
}
print(f"   Code generation score: {eval_code['average_score']:.3f}")

result_multi['evaluations'] = evaluation_results

# Calculate robustness
scores = [eval_results['average_score'] for eval_results in evaluation_results.values()]
robustness = np.std(scores)
avg_score = np.mean(scores)

with open("experiments/multi_domain_crpo_reward.json", "w") as f:
    json.dump(result_multi, f, indent=2)

print("\n" + "=" * 70)
print("MULTI-DOMAIN RESULTS")
print("=" * 70)
print(f"Average score across domains: {avg_score:.3f}")
print(f"Robustness (std dev): {robustness:.4f}")
print(f"Results saved to: experiments/multi_domain_crpo_reward.json")
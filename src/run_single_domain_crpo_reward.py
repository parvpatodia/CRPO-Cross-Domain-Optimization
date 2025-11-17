import json
import os
import numpy as np
from dotenv import load_dotenv
from crpo_baseline import CRPOBaseline
from data_loader import DataLoader
from evaluation_reward import EvaluatorWithRewardModel

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

crpo = CRPOBaseline(api_key)
evaluator = EvaluatorWithRewardModel(api_key)
loader = DataLoader()

print("=" * 70)
print("SINGLE-DOMAIN CRPO WITH REWARD MODEL")
print("=" * 70)

# Load HelpSteer2
helpsteer2 = loader.load_helpsteer2()
print(f"\nLoaded HelpSteer2 with {len(helpsteer2)} reference examples")

all_results = {}

# DOMAIN 1: MATH
print("\n" + "=" * 70)
print("DOMAIN 1: MATHEMATICS (GSM8K)")
print("=" * 70)

task_math = "Solve grade-school math word problems"
result_math = crpo.optimize(task_math, helpsteer2, domain='math')
all_results['math'] = result_math

# Evaluate optimized prompt on GSM8K
print("\nEvaluating optimized prompt on GSM8K test set...")
gsm8k_test = loader.load_gsm8k('test')[:100]
eval_math = evaluator.evaluate_dataset(
    result_math['optimized_prompt'] + "\n\nQuestion: {question}\n\nAnswer:",
    gsm8k_test,
    'math',
    max_examples=100
)
all_results['math']['evaluation'] = {
    'average_score': eval_math['average_score'],
    'std_dev': eval_math['std_dev'],
    'api_calls_used': eval_math['api_calls_used']
}
print(f"Math average score: {eval_math['average_score']:.3f}")

# DOMAIN 2: REASONING
print("\n" + "=" * 70)
print("DOMAIN 2: REASONING (BBH Navigate)")
print("=" * 70)

task_reasoning = "Solve logical reasoning and spatial navigation tasks"
result_reasoning = crpo.optimize(task_reasoning, helpsteer2, domain='reasoning')
all_results['reasoning'] = result_reasoning

print("\nEvaluating optimized prompt on BBH Navigate test set...")
_, bbh_test = loader.load_bbh('navigate')
bbh_test = bbh_test[:50]
eval_reasoning = evaluator.evaluate_dataset(
    result_reasoning['optimized_prompt'] + "\n\nQuestion: {question}\n\nAnswer:",
    bbh_test,
    'reasoning',
    max_examples=50
)
all_results['reasoning']['evaluation'] = {
    'average_score': eval_reasoning['average_score'],
    'std_dev': eval_reasoning['std_dev'],
    'api_calls_used': eval_reasoning['api_calls_used']
}
print(f"Reasoning average score: {eval_reasoning['average_score']:.3f}")

# DOMAIN 3: FACT VERIFICATION
print("\n" + "=" * 70)
print("DOMAIN 3: FACT VERIFICATION (LIAR)")
print("=" * 70)

task_fact = "Verify the truthfulness of statements"
result_fact = crpo.optimize(task_fact, helpsteer2, domain='fact_verification')
all_results['fact'] = result_fact

print("\nEvaluating optimized prompt on LIAR test set...")
liar_test = loader.load_liar('test')[:50]
eval_fact = evaluator.evaluate_dataset(
    result_fact['optimized_prompt'] + "\n\nStatement: {question}\n\nAnswer:",
    liar_test,
    'fact_verification',
    max_examples=50
)
all_results['fact']['evaluation'] = {
    'average_score': eval_fact['average_score'],
    'std_dev': eval_fact['std_dev'],
    'api_calls_used': eval_fact['api_calls_used']
}
print(f"Fact verification average score: {eval_fact['average_score']:.3f}")

# DOMAIN 4: CODE
print("\n" + "=" * 70)
print("DOMAIN 4: CODE GENERATION (HumanEval)")
print("=" * 70)

task_code = "Generate correct Python code"
result_code = crpo.optimize(task_code, helpsteer2, domain='code')
all_results['code'] = result_code

print("\nEvaluating optimized prompt on HumanEval test set...")
code_data = loader.load_humaneval(50)
code_test = code_data[25:]
eval_code = evaluator.evaluate_dataset(
    result_code['optimized_prompt'] + "\n\nProblem: {question}\n\nSolution:",
    code_test,
    'code',
    max_examples=25
)
all_results['code']['evaluation'] = {
    'average_score': eval_code['average_score'],
    'std_dev': eval_code['std_dev'],
    'api_calls_used': eval_code['api_calls_used']
}
print(f"Code generation average score: {eval_code['average_score']:.3f}")

# Save results
os.makedirs("experiments", exist_ok=True)
with open("experiments/single_domain_crpo_reward.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 70)
print("SINGLE-DOMAIN CRPO COMPLETE")
print(f"Total API calls used: {crpo.api_calls + evaluator.api_calls}")
print(f"Results saved to: experiments/single_domain_crpo_reward.json")
print("=" * 70)

# Print summary
print("\nSUMMARY:")
for domain, results in all_results.items():
    score = results.get('evaluation', {}).get('average_score', 0)
    print(f"{domain:15} | Score: {score:.3f}")
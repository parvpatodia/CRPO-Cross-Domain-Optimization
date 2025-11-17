from groq import Groq
import json
import os
from typing import List, Dict
import time
import re

class Evaluator:
    """Evaluate prompts across all domains using smart domain-specific heuristics"""
    
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.results = {}
        self.api_calls = 0
    
    def evaluate_dataset(self, prompt_template: str, dataset: List[Dict], 
                        domain: str, max_examples=None) -> Dict:
        """Evaluate prompt on a dataset"""
        
        if max_examples:
            dataset = dataset[:max_examples]
        
        correct = 0
        total = len(dataset)
        detailed_results = []
        
        print(f"\nEvaluating {domain}...")
        print(f"Examples: {total}")
        
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                print(f"  Progress: {i}/{total}")
            
            # Format prompt with example
            full_prompt = prompt_template.format(question=example['prompt'])
            
            try:
                # Get model response
                response = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                self.api_calls += 1
                model_response = response.choices[0].message.content
                
                # Check correctness using domain-specific heuristics
                is_correct = self.check_correctness_smart(
                    model_response,
                    example['answer'],
                    domain
                )
                
                if is_correct:
                    correct += 1
                
                detailed_results.append({
                    'example_id': example['id'],
                    'prompt': example['prompt'][:100],
                    'expected': str(example['answer'])[:100],
                    'model_response': model_response[:100],
                    'correct': is_correct
                })
                
                time.sleep(0.05)
            
            except Exception as e:
                print(f"Error on example {i}: {e}")
                detailed_results.append({
                    'example_id': example['id'],
                    'error': str(e)
                })
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'domain': domain,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'api_calls_used': self.api_calls,
            'details': detailed_results
        }
    
    @staticmethod
    def check_correctness_smart(response: str, expected, domain: str) -> bool:
        """Smart domain-specific correctness checking"""
        
        response_clean = str(response).strip().lower()
        expected_clean = str(expected).strip().lower()
        
        if domain == 'math':
            """
            For math: Extract final number from response and compare with expected.
            Llama tends to end with the final answer clearly.
            """
            # Get the actual answer from expected (usually after "=")
            if '=' in expected_clean:
                expected_answer = expected_clean.split('=')[-1].strip()
            else:
                expected_answer = expected_clean
            
            # Extract numbers from expected
            expected_nums = re.findall(r'-?\d+\.?\d*', expected_answer)
            if not expected_nums:
                return False
            
            expected_value = float(expected_nums[0])
            
            # Extract ALL numbers from response
            response_nums = re.findall(r'-?\d+\.?\d*', response_clean)
            if not response_nums:
                return False
            
            # Check last few numbers (model often states answer at end)
            for response_num_str in response_nums[-3:]:
                try:
                    response_value = float(response_num_str)
                    # Allow 5% tolerance for rounding/intermediate steps
                    if abs(response_value - expected_value) < abs(expected_value * 0.05) + 0.1:
                        return True
                except ValueError:
                    continue
            
            return False
        
        elif domain == 'reasoning':
            """
            For reasoning: Check if answer matches expected (true/false/yes/no).
            Llama is good at this - just look for the words.
            """
            # Normalize the expected answer
            if any(word in expected_clean for word in ['true', 'yes', 'correct', '1']):
                expected_answer = 'true'
            elif any(word in expected_clean for word in ['false', 'no', '0']):
                expected_answer = 'false'
            else:
                return False
            
            # Check if response contains the right answer
            if expected_answer == 'true':
                return any(word in response_clean for word in ['true', 'yes', 'correct'])
            else:
                return any(word in response_clean for word in ['false', 'no', 'incorrect'])
        
        elif domain == 'fact_verification':
            """
            For fact verification: Check if response judgment matches expected.
            Llama is decent at this - look for truthfulness keywords.
            """
            # Normalize expected
            if any(word in expected_clean for word in ['true', 'correct', 'supported', 'verified', '4', '3']):
                expected_category = 'true'
            elif any(word in expected_clean for word in ['false', 'incorrect', 'contradicted', '0', '1']):
                expected_category = 'false'
            else:
                expected_category = 'partial'
            
            # Check response
            response_has_true = any(word in response_clean for word in ['true', 'correct', 'supported', 'verified'])
            response_has_false = any(word in response_clean for word in ['false', 'incorrect', 'contradicted', 'wrong'])
            response_has_partial = any(word in response_clean for word in ['partial', 'mixed', 'some', 'half'])
            
            # Match on most prominent keyword
            if expected_category == 'true':
                return response_has_true and not response_has_false
            elif expected_category == 'false':
                return response_has_false and not response_has_true
            else:
                return response_has_partial or (not response_has_true and not response_has_false)
        
        elif domain == 'code':
            """
            For code: Check if response looks like real Python code.
            We can't execute, so we check for valid syntax.
            """
            import ast
            
            try:
                # Try to parse as Python
                ast.parse(response_clean)
                
                # Also check for code patterns
                has_def = 'def ' in response_clean
                has_return = 'return' in response_clean
                
                # Valid code should have at least one of these
                return has_def or has_return
            
            except SyntaxError:
                return False
        
        return False

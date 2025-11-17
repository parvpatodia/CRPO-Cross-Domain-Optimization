from groq import Groq
import json
import os
from typing import List, Dict
import time
import numpy as np
from reward_scorer import RewardModelScorer

class EvaluatorWithRewardModel:
    """Evaluate prompts using a reward model instead of accuracy heuristics"""
    
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.reward_scorer = RewardModelScorer()
        self.results = {}
        self.api_calls = 0
    
    def evaluate_dataset(self, prompt_template: str, dataset: List[Dict], 
                        domain: str, max_examples=None) -> Dict:
        """Evaluate prompt on a dataset using reward model scoring"""
        
        if max_examples:
            dataset = dataset[:max_examples]
        
        scores = []
        detailed_results = []
        
        print(f"\nEvaluating {domain}...")
        print(f"Examples: {len(dataset)}")
        
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(dataset)}")
            
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
                
                # Score using reward model
                score = self.reward_scorer.score_response(full_prompt, model_response)
                scores.append(score)
                
                detailed_results.append({
                    'example_id': example['id'],
                    'prompt': example['prompt'][:100],
                    'response': model_response[:100],
                    'score': score
                })
                
                # Rate limiting
                time.sleep(0.05)
            
            except Exception as e:
                print(f"Error on example {i}: {e}")
                detailed_results.append({
                    'example_id': example['id'],
                    'error': str(e)
                })
        
        # Calculate statistics
        average_score = np.mean(scores) if scores else 0
        std_dev = np.std(scores) if scores else 0
        
        return {
            'domain': domain,
            'average_score': average_score,
            'std_dev': std_dev,
            'min_score': np.min(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'num_examples': len(scores),
            'api_calls_used': self.api_calls,
            'details': detailed_results
        }
import json
import random
from typing import List, Dict

class DataLoader:
    """Load and preprocess all datasets"""
    
    @staticmethod
    def load_gsm8k(split='train', n_samples=None):
        """Load GSM8K math problems"""
        with open(f"data/raw/gsm8k/{split}.json", "r") as f:
            data = json.load(f)
        
        if split == 'train':
            data = random.sample(data, min(500, len(data)))
        elif split == 'test':
            data = random.sample(data, min(1000, len(data)))
        
        # Standardize format
        processed = []
        for item in data:
            processed.append({
                'id': f"gsm8k_{len(processed)}",
                'prompt': item['question'],
                'answer': item['answer'],
                'domain': 'math'
            })
        
        return processed
    
    @staticmethod
    def load_bbh(task='navigate'):
        """Load BBH tasks"""
        # Map task names to actual filenames
        task_map = {
            'navigate': 'navigate',
            'boolean_expressions': 'boolean',
            'boolean': 'boolean'
        }
        filename = task_map.get(task, task)
        with open(f"data/raw/bbh/{filename}.json", "r") as f:
            data = json.load(f)
        
        # Split 70-30
        random.seed(42)
        random.shuffle(data)
        split_idx = int(len(data) * 0.7)
        
        train = data[:split_idx]
        test = data[split_idx:]
        
        def process_split(items, split_name):
            processed = []
            for item in items:
                processed.append({
                    'id': f"bbh_{task}_{len(processed)}",
                    'prompt': item['input'],
                    'answer': item['target'],
                    'domain': 'reasoning'
                })
            return processed
        
        return process_split(train, 'train'), process_split(test, 'test')
    
    @staticmethod
    def load_liar(split='test'):
        """Load LIAR fact verification"""
        with open(f"data/raw/liar/{split}.json", "r") as f:
            data = json.load(f)
        
        # Map LIAR labels
        label_map = {
            0: 'false',      # pants-fire
            1: 'false',      # false
            2: 'half-true',  # half-true
            3: 'true',       # mostly-true
            4: 'true'        # true
        }
        
        processed = []
        for item in data:
            # Get the label - it might be 'label' or encoded as int
            label = item.get('label', item.get('truthfulness', 0))
            
            # Map to human-readable label
            if isinstance(label, int):
                label = label_map.get(label, 'unknown')
            else:
                label = str(label).lower()
            
            processed.append({
                'id': f"liar_{len(processed)}",
                'prompt': item.get('statement', ''),
                'answer': label,  # Now this is a string like 'true', 'false', etc.
                'domain': 'fact_verification'
            })
        
        return processed

    
    @staticmethod
    def load_humaneval(n_samples=50):
        """Load code generation tasks"""
        with open("data/raw/humaneval/samples.json", "r") as f:
            data = json.load(f)
        
        processed = []
        for item in data[:n_samples]:
            processed.append({
                'id': f"code_{len(processed)}",
                'prompt': item['prompt'],
                'answer': item['canonical_solution'],
                'domain': 'code'
            })
        
        return processed
    
    @staticmethod
    def load_helpsteer2():
        """Load reference examples for CRPO optimization"""
        with open("data/raw/helpsteer2/full.json", "r") as f:
            data = json.load(f)
        
        # Extract quality scores (if available)
        processed = []
        for item in data:
            processed.append({
                'id': f"help_{len(processed)}",
                'prompt': item.get('prompt', ''),
                'response': item.get('response', ''),
                'quality_score': item.get('helpfulness', 5)  # normalize
            })
        
        return processed

# Quick test
if __name__ == "__main__":
    loader = DataLoader()
    
    print("Loading GSM8K train...")
    gsm8k_train = loader.load_gsm8k('train')
    print(f"Loaded {len(gsm8k_train)} examples")
    print(f"Sample: {gsm8k_train[0]}")
    
    print("\nLoading LIAR...")
    liar = loader.load_liar('test')
    print(f"Loaded {len(liar)} examples")
    print(f"Sample: {liar[0]}")
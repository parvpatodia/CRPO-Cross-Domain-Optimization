import json
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from src.config import (
    GSM8K_DIR, BBH_DIR, LIAR_DIR, HUMANEVAL_DIR, HELPSTEER2_DIR
)

class DataLoader:
    """Load and preprocess all datasets using centralized configuration."""
    
    @staticmethod
    def _load_json(file_path: Path) -> List[Dict[str, Any]]:
        """Helper to load JSON data with error handling."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {file_path}")
            return []

    @staticmethod
    def load_gsm8k(split: str = 'train', n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load GSM8K math problems."""
        file_path = GSM8K_DIR / f"{split}.json"
        data = DataLoader._load_json(file_path)
        
        if not data:
            return []

        if split == 'train':
            sample_size = min(500, len(data)) if n_samples is None else min(n_samples, len(data))
            data = random.sample(data, sample_size)
        elif split == 'test':
            sample_size = min(1000, len(data)) if n_samples is None else min(n_samples, len(data))
            data = random.sample(data, sample_size)
        
        # Standardize format
        processed = []
        for i, item in enumerate(data):
            processed.append({
                'id': f"gsm8k_{i}",
                'prompt': item.get('question', ''),
                'answer': item.get('answer', ''),
                'domain': 'math'
            })
        
        return processed
    
    @staticmethod
    def load_bbh(task: str = 'navigate') -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load BBH tasks."""
        # Map task names to actual filenames
        task_map = {
            'navigate': 'navigate',
            'boolean_expressions': 'boolean',
            'boolean': 'boolean'
        }
        filename = task_map.get(task, task)
        file_path = BBH_DIR / f"{filename}.json"
        
        data = DataLoader._load_json(file_path)
        if not data:
            return [], []
        
        # Split 70-30
        random.seed(42)
        random.shuffle(data)
        split_idx = int(len(data) * 0.7)
        
        train = data[:split_idx]
        test = data[split_idx:]
        
        def process_split(items: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
            processed = []
            for i, item in enumerate(items):
                processed.append({
                    'id': f"bbh_{task}_{i}",
                    'prompt': item.get('input', ''),
                    'answer': item.get('target', ''),
                    'domain': 'reasoning'
                })
            return processed
        
        return process_split(train, 'train'), process_split(test, 'test')
    
    @staticmethod
    def load_liar(split: str = 'test') -> List[Dict[str, Any]]:
        """Load LIAR fact verification."""
        file_path = LIAR_DIR / f"{split}.json"
        data = DataLoader._load_json(file_path)
        
        if not data:
            return []
        
        # Map LIAR labels
        label_map = {
            0: 'false',      # pants-fire
            1: 'false',      # false
            2: 'half-true',  # half-true
            3: 'true',       # mostly-true
            4: 'true'        # true
        }
        
        processed = []
        for i, item in enumerate(data):
            # Get the label - it might be 'label' or encoded as int
            label = item.get('label', item.get('truthfulness', 0))
            
            # Map to human-readable label
            if isinstance(label, int):
                label = label_map.get(label, 'unknown')
            else:
                label = str(label).lower()
            
            processed.append({
                'id': f"liar_{i}",
                'prompt': item.get('statement', ''),
                'answer': label,
                'domain': 'fact_verification'
            })
        
        return processed

    @staticmethod
    def load_humaneval(n_samples: int = 50) -> List[Dict[str, Any]]:
        """Load code generation tasks."""
        file_path = HUMANEVAL_DIR / "samples.json"
        data = DataLoader._load_json(file_path)
        
        if not data:
            return []
        
        processed = []
        for i, item in enumerate(data[:n_samples]):
            processed.append({
                'id': f"code_{i}",
                'prompt': item.get('prompt', ''),
                'answer': item.get('canonical_solution', ''),
                'domain': 'code'
            })
        
        return processed
    
    @staticmethod
    def load_helpsteer2() -> List[Dict[str, Any]]:
        """Load reference examples for CRPO optimization."""
        file_path = HELPSTEER2_DIR / "full.json"
        data = DataLoader._load_json(file_path)
        
        if not data:
            return []
        
        # Extract quality scores (if available)
        processed = []
        for i, item in enumerate(data):
            processed.append({
                'id': f"help_{i}",
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
    if gsm8k_train:
        print(f"Sample: {gsm8k_train[0]}")
    
    print("\nLoading LIAR...")
    liar = loader.load_liar('test')
    print(f"Loaded {len(liar)} examples")
    if liar:
        print(f"Sample: {liar[0]}")
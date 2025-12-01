from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from typing import List, Tuple, Optional
from src.config import REWARD_MODEL_NAME

class RewardModelScorer:
    """Score responses using a pre-trained reward model (like the paper does)"""
    
    def __init__(self, model_name: str = REWARD_MODEL_NAME):
        """
        Initialize reward model. This model scores prompt-response pairs on quality.
        Similar to what the paper uses (ArmoRM-Llama3-8B-v0.1)
        """
        print(f"Loading reward model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            print(f"Reward model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading reward model: {e}")
            print("Using fallback simple scoring (may be less accurate)")
            self.model = None
    
    def score_response(self, prompt: str, response: str) -> float:
        """
        Score a prompt-response pair.
        Returns a score typically between 0 and 1 (or -1 to 1 depending on model).
        """
        
        if self.model is None:
            return self._fallback_score(response)
        
        try:
            # Prepare input (prompt followed by response)
            text = f"{prompt} {response}"
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits[0, 0].item()  # Extract scalar score
            
            # Normalize to [0, 1] range
            # Most reward models output in range [-5, 5] or similar
            normalized_score = (score + 5) / 10  # Adjust based on model's typical output
            normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
            
            return normalized_score
        
        except Exception as e:
            print(f"Error scoring response: {e}")
            return self._fallback_score(response)
    
    @staticmethod
    def _fallback_score(response: str) -> float:
        """Fallback heuristic if reward model fails"""
        # Simple heuristic: longer, more detailed responses score higher
        length_score = min(len(response) / 500, 1.0)
        
        # Has reasoning keywords
        keywords = ['step', 'therefore', 'because', 'first', 'second', 'finally']
        has_reasoning = sum(1 for kw in keywords if kw in response.lower())
        reasoning_score = min(has_reasoning / 3, 1.0)
        
        return 0.6 * length_score + 0.4 * reasoning_score
    
    def score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Score multiple prompt-response pairs"""
        scores = []
        for prompt, response in zip(prompts, responses):
            score = self.score_response(prompt, response)
            scores.append(score)
        return scores

# Test the scorer
if __name__ == "__main__":
    scorer = RewardModelScorer()
    
    # Test
    test_prompt = "What is 2+2?"
    test_response = "2+2 equals 4"
    
    score = scorer.score_response(test_prompt, test_response)
    print(f"Test score: {score:.3f}")
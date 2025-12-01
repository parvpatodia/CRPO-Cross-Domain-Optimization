from datasets import load_dataset
import json
import os
from src.config import GSM8K_DIR

def download_gsm8k():
    """Download and save GSM8K dataset."""
    print("Downloading GSM8K...")
    try:
        gsm8k = load_dataset("openai/gsm8k", "main")
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")
        return

    # Ensure directory exists
    GSM8K_DIR.mkdir(parents=True, exist_ok=True)

    train_data = gsm8k['train']
    test_data = gsm8k['test']

    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # Save as JSON
    try:
        with open(GSM8K_DIR / "train.json", "w") as f:
            json.dump([dict(item) for item in train_data], f)
        
        with open(GSM8K_DIR / "test.json", "w") as f:
            json.dump([dict(item) for item in test_data], f)
            
        print(f"GSM8K saved to {GSM8K_DIR}")
    except IOError as e:
        print(f"Error saving files: {e}")

if __name__ == "__main__":
    download_gsm8k()

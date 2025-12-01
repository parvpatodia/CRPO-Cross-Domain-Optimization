"""
Note: There's a known issue with datasets==2.14.0 and script-based datasets
that causes LocalFileSystem caching errors. This script uses a workaround.
"""
from datasets import load_dataset_builder
import json
import os
import random
import pyarrow as pa
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR

print("Downloading LIAR...")

def load_liar_split(cache_dir, split_name):
    """Load LIAR split by reading Arrow files directly from cache"""
    # Find Arrow files in cache
    arrow_files = []
    
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.arrow') and split_name in file.lower():
                arrow_files.append(os.path.join(root, file))
    
    if not arrow_files:
        # Try to find any arrow files with the split name
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith('.arrow'):
                    full_path = os.path.join(root, file)
                    if split_name in os.path.basename(full_path).lower():
                        arrow_files.append(full_path)
    
    if arrow_files:
        arrow_file = arrow_files[0]
        print(f"  Reading {split_name} from {os.path.basename(arrow_file)}...")
        
        # Read Arrow file
        try:
            mmap = pa.memory_map(arrow_file, 'r')
            reader = pa.ipc.RecordBatchFileReader(mmap)
            table = reader.read_all()
            mmap.close()
        except:
            # Try as stream instead
            with open(arrow_file, 'rb') as f:
                reader = pa.ipc.RecordBatchStreamReader(f)
                table = reader.read_all()
        
        examples = []
        for i in range(len(table)):
            row = {}
            for col in table.column_names:
                val = table[col][i].as_py()
                row[col] = val
            examples.append(row)
        return examples
    else:
        raise RuntimeError(f"Could not find {split_name} split files.")

# Download and prepare dataset once
print("Downloading and preparing LIAR dataset...")
builder = load_dataset_builder("liar", "liar")
builder.download_and_prepare(download_mode="force_redownload")
cache_dir = builder._cache_dir

# Load all splits from the same cache
print("Loading train split...")
train_data = load_liar_split(cache_dir, "train")

print("Loading test split...")
test_data = load_liar_split(cache_dir, "test")

# Combine train + test
all_data = train_data + test_data

print(f"Total LIAR examples: {len(all_data)}")

# Sample 800 for optimization, keep 500 for test
# Stratify by label to maintain class balance
random.seed(42)
random.shuffle(all_data)

# Simple split (in production, use stratified split)
opt_data = all_data[:800]
test_data = all_data[800:1300]

print(f"Optimization examples: {len(opt_data)}")
print(f"Test examples: {len(test_data)}")

# Ensure directory exists
liar_dir = RAW_DATA_DIR / "liar"
liar_dir.mkdir(parents=True, exist_ok=True)

with open(liar_dir / "optimization.json", "w") as f:
    json.dump(opt_data, f, indent=2)

with open(liar_dir / "test.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"LIAR saved to {liar_dir}")
"""
Note: There's a known issue with datasets==2.14.0 and script-based datasets
that causes LocalFileSystem caching errors. This script uses a workaround.
"""
from huggingface_hub import hf_hub_download
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR

print("Downloading BBH...")

def load_bbh_config(config_name):
    """
    Load BBH config by downloading raw data files directly from HuggingFace Hub.
    This bypasses the datasets library cache issues.
    """
    print(f"  Downloading {config_name} config...")
    
    # Download the dataset repository files
    # BBH stores data in JSON format in the repository
    try:
        # Try to download the data file directly
        # The structure may vary, so we'll try common patterns
        data_file = hf_hub_download(
            repo_id="maveriq/bigbenchhard",
            filename=f"{config_name}.json",
            repo_type="dataset"
        )
        
        # Read the JSON file
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Handle different data structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'train' in data:
            return data['train']
        else:
            return [data]
            
    except Exception as e:
        print(f"  Direct download not available (data is script-generated), using datasets library...")
        # Use datasets library - the data generates successfully, we just need to read it
        from datasets import load_dataset_builder
        import pyarrow as pa
        
        builder = load_dataset_builder("maveriq/bigbenchhard", config_name)
        builder.download_and_prepare(download_mode="force_redownload")
        
        # The data was generated, now read the Arrow files directly
        cache_dir = builder._cache_dir
        
        # Find Arrow files in cache
        arrow_files = []
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith('.arrow') and 'train' in file.lower():
                    arrow_files.append(os.path.join(root, file))
        
        if not arrow_files:
            # Try to find any arrow files
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith('.arrow'):
                        arrow_files.append(os.path.join(root, file))
        
        if arrow_files:
            # Read the first Arrow file found
            arrow_file = arrow_files[0]
            print(f"  Reading from {os.path.basename(arrow_file)}...")
            
            # Read Arrow file using memory map
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
            raise RuntimeError(f"Could not find generated data files for {config_name}. Cache may be corrupted.")

# Load specific configs
print("Loading navigate config...")
navigate = load_bbh_config("navigate")

print("Loading boolean_expressions config...")
boolean = load_bbh_config("boolean_expressions")

print(f"Navigate examples: {len(navigate)}")
print(f"Boolean examples: {len(boolean)}")

# Ensure directory exists
bbh_dir = RAW_DATA_DIR / "bbh"
bbh_dir.mkdir(parents=True, exist_ok=True)

# Save
with open(bbh_dir / "navigate.json", "w") as f:
    json.dump(navigate, f, indent=2)

with open(bbh_dir / "boolean.json", "w") as f:
    json.dump(boolean, f, indent=2)

print(f"BBH saved to {bbh_dir}")
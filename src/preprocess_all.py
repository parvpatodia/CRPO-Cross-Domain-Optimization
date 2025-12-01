import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader
from src.config import PROCESSED_DATA_DIR

loader = DataLoader()

print("=" * 50)
print("PREPROCESSING ALL DATASETS")
print("=" * 50)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# GSM8K
print("\n1. GSM8K...")
gsm8k_train = loader.load_gsm8k('train')
gsm8k_test = loader.load_gsm8k('test')
with open(PROCESSED_DATA_DIR / "gsm8k_train.json", "w") as f:
    json.dump(gsm8k_train, f)
with open(PROCESSED_DATA_DIR / "gsm8k_test.json", "w") as f:
    json.dump(gsm8k_test, f)
print(f"   Train: {len(gsm8k_train)}, Test: {len(gsm8k_test)}")

# BBH Navigate
print("\n2. BBH Navigate...")
bbh_nav_train, bbh_nav_test = loader.load_bbh('navigate')
with open(PROCESSED_DATA_DIR / "bbh_navigate_train.json", "w") as f:
    json.dump(bbh_nav_train, f)
with open(PROCESSED_DATA_DIR / "bbh_navigate_test.json", "w") as f:
    json.dump(bbh_nav_test, f)
print(f"   Train: {len(bbh_nav_train)}, Test: {len(bbh_nav_test)}")

# BBH Boolean
print("\n3. BBH Boolean...")
bbh_bool_train, bbh_bool_test = loader.load_bbh('boolean_expressions')
with open(PROCESSED_DATA_DIR / "bbh_boolean_train.json", "w") as f:
    json.dump(bbh_bool_train, f)
with open(PROCESSED_DATA_DIR / "bbh_boolean_test.json", "w") as f:
    json.dump(bbh_bool_test, f)
print(f"   Train: {len(bbh_bool_train)}, Test: {len(bbh_bool_test)}")

# LIAR
print("\n4. LIAR...")
# Note: load_liar only supports 'test' in the current data_loader implementation
# If optimization split is needed, data_loader needs update. 
# For now, we'll just save the test split which is what's available.
liar_test = loader.load_liar('test')
with open(PROCESSED_DATA_DIR / "liar_test.json", "w") as f:
    json.dump(liar_test, f)
print(f"   Test: {len(liar_test)}")

# HumanEval
print("\n5. HumanEval...")
code = loader.load_humaneval(50)
code_train = code[:25]
code_test = code[25:]
with open(PROCESSED_DATA_DIR / "code_train.json", "w") as f:
    json.dump(code_train, f)
with open(PROCESSED_DATA_DIR / "code_test.json", "w") as f:
    json.dump(code_test, f)
print(f"   Train: {len(code_train)}, Test: {len(code_test)}")

# HelpSteer2
print("\n6. HelpSteer2...")
helpsteer2 = loader.load_helpsteer2()
with open(PROCESSED_DATA_DIR / "helpsteer2_full.json", "w") as f:
    json.dump(helpsteer2, f)
print(f"   Total: {len(helpsteer2)}")

print("\n" + "=" * 50)
print("ALL DATASETS PREPROCESSED SUCCESSFULLY")
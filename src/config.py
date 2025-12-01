import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset Specific Paths
GSM8K_DIR = RAW_DATA_DIR / "gsm8k"
BBH_DIR = RAW_DATA_DIR / "bbh"
LIAR_DIR = RAW_DATA_DIR / "liar"
HUMANEVAL_DIR = RAW_DATA_DIR / "humaneval"
HELPSTEER2_DIR = RAW_DATA_DIR / "helpsteer2"

# Experiment Paths
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
DEFAULT_MODEL = "llama-3.1-8b-instant"
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large"
TEMPERATURE = 0.7
MAX_TOKENS = 100

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENTS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

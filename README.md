# CRPO Cross-Domain Optimization

This project implements **Contrastive Reasoning Prompt Optimization (CRPO)**, a technique for generating robust, high-quality prompts that generalize across multiple diverse domains (Mathematics, Reasoning, Fact Verification, and Code Generation).

The system uses the **Groq API (Llama 3.1)** to analyze high- and low-quality examples from different domains, extract universal reasoning patterns, and generate a single optimized prompt that performs well across all tasks.

## Project Structure

```text
CRPO-Cross-Domain-Optimization/
├── data/               # Raw and processed datasets (GSM8K, BBH, LIAR, etc.)
├── experiments/        # Experiment logs and configuration files
├── results/            # Analysis results, tables, and visualizations
├── src/                # Source code
│   ├── crpo_multidomain.py   # Core Multi-Domain CRPO algorithm
│   ├── crpo_baseline.py      # Single-domain CRPO implementation
│   ├── evaluation_reward.py  # Evaluation using reward models
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── config.py             # Centralized configuration
│   └── ...                   # Other utility scripts
├── test_groq.py        # Script to verify Groq API connection
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Features

*   **Multi-Domain Optimization**: Optimizes a single prompt to work effectively on Math (GSM8K), Reasoning (BBH), Fact Verification (LIAR), and Code (HumanEval) simultaneously.
*   **Contrastive Reasoning**: Analyzes the difference between high-quality and low-quality responses to derive effective prompting strategies.
*   **Reward-Based Evaluation**: Uses a reward model approach to evaluate the quality of generated responses.
*   **Comprehensive Analysis**: Includes scripts for zero-shot and few-shot baselines, as well as detailed comparative analysis and visualization.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd CRPO-Cross-Domain-Optimization
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    You need a Groq API key to run the optimization and evaluation.
    ```bash
    export GROQ_API_KEY="your_groq_api_key_here"
    ```
    Alternatively, create a `.env` file in the root directory:
    ```text
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

### 1. Data Preparation
The project includes scripts to download and preprocess the necessary datasets.
```bash
python src/download_gsm8k.py
python src/download_bbh.py
python src/download_liar.py
# Or run the all-in-one preprocessor
python src/preprocess_all.py
```

### 2. Running Baselines
Establish baseline performance for comparison.
```bash
# Zero-shot baseline
python src/run_zero_shot_reward.py

# Few-shot baseline
python src/run_few_shot_reward.py
```

### 3. Running Optimization
Run the CRPO optimization process.

**Single-Domain Optimization** (optimizes for each domain individually):
```bash
python src/run_single_domain_crpo_reward.py
```

**Multi-Domain Optimization** (optimizes for all domains simultaneously):
```bash
python src/run_multidomain_crpo_reward.py
```

### 4. Analysis & Visualization
Generate tables and figures to compare the methods.
```bash
# Generate analysis table (CSV/JSON)
python src/create_final_analysis_reward.py

# Generate visualizations (plots)
python src/create_visualizations_reward.py
```

## Methodology

The **Multi-Domain CRPO** process works as follows:
1.  **Retrieve Examples**: Selects high and low-quality examples from a reference dataset (e.g., HelpSteer2) that act as proxies for different domains.
2.  **Contrastive Reasoning**: The model analyzes these examples to identify prompt properties that distinguish high-quality outputs from low-quality ones across *all* domains.
3.  **Prompt Generation**: A new prompt is generated that incorporates these universal insights, aiming for robustness and generalization.
4.  **Evaluation**: The optimized prompt is tested on unseen test sets from GSM8K, BBH, LIAR, and HumanEval.

## Results

Results are saved in the `results/` directory. The analysis script produces a comparison table showing Zero-Shot, Few-Shot, Single-Domain CRPO, and Multi-Domain CRPO performance across all tested domains.

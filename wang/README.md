# Financial Analysis Projects

This directory contains scripts and notebooks for financial analysis using Machine Learning and LLMs.

## 1. Llama-3 Stock Movement Predictor

This project fine-tunes a **Llama-3-8b** model to predict stock price movements (UP/DOWN) based on financial news headlines.

### Pipeline Overview

The workflow consists of 4 steps, available as both Jupyter Notebooks and Python scripts:

1.  **Data Collection** (`01_data_collection.py` / `Untitled.ipynb`)
    *   Crawls Google News for major tech stocks (NVDA, TSM, AMD, etc.).
    *   Downloads historical prices via `yfinance`.
    *   Labels news as UP/DOWN based on next-day returns.
    *   Output: `my_custom_fin_dataset_2025.csv`

2.  **Data Preprocessing** (`02_data_preprocessing.py` / `Untitled1.ipynb`)
    *   Converts the CSV into a JSONL format for Llama-3.
    *   Generates synthetic "Chain of Thought" reasoning for training.
    *   Output: `llama3_finetune_data.jsonl`

3.  **Model Fine-tuning** (`03_model_finetuning.py` / `Untitled2.ipynb`)
    *   Uses `unsloth` to fine-tune `unsloth/llama-3-8b-bnb-4bit`.
    *   Saves the adapter checkpoint to `llama3_financial_analyst_checkpoint`.

4.  **Inference** (`04_inference.py` / `Untitled3.ipynb`)
    *   Loads the fine-tuned model.
    *   Predicts stock movement for new headlines.

### ⚠️ CRITICAL WARNING: Data Leakage

**Do not use the current pipeline for backtesting on 2025 data.**

*   **Issue**: The training dataset currently includes data up to late 2025.
*   **Consequence**: Testing on 2025 data will result in overfitting (the model has already "seen" the answers).
*   **Fix Required**: Before running a backtest, modify `02_data_preprocessing.py` to split the data chronologically:
    *   **Train**: Data < 2025-01-01
    *   **Test**: Data >= 2025-01-01

---

## 2. Fed FOMC Analysis

*   **Script**: `data_pipeline.py`
*   **Purpose**: Crawls Federal Reserve FOMC statements (2020-2025) and correlates them with market volatility data (VIX, TLT).
*   **Output**: `fed_project_data.csv`

# BanglaSocialBench 🇧🇩

[![Hugging Face Dataset](https://img.shields.io/badge/🤗_Hugging_Face-Dataset-blue.svg)](https://huggingface.co/datasets/sijantanvir/BanglaSocialBench)

**BanglaSocialBench** is an evaluation framework designed to test the cultural and sociopragmatic alignment of Large Language Models in the context of the Bangla language. 

This repository provides evaluation scripts to reproduce our baseline results or test your own models against the benchmark.

## 📊 Dataset Access
The full dataset is hosted on Hugging Face. You do not need to download any local CSVs to run this evaluation pipeline; the scripts pull directly from the hub.

*   **Dataset Link:** [sijantanvir/BanglaSocialBench](https://huggingface.co/datasets/sijantanvir/BanglaSocialBench)

## 🛠️ Repository Structure
This repository contains the essential scripts needed to prompt models, parse their responses, and grade their sociopragmatic accuracy.

```text
BanglaSocialBench/
├── src/                                  
│   ├── llm_clients.py        # API wrappers (OpenAI, Gemini, OpenRouter, TogetherAI)
│   ├── parsing.py            # Text normalization and answer extraction logic
│   └── prompts.py            # System prompts and few-shot templates
├── scripts/                              
│   └── 01_evaluate_models.py # Main execution loop for scoring models
├── .env.example              # Template for API keys
└── requirements.txt          # Python dependencies

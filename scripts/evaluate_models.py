import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from datasets import load_dataset

from src.llm_clients import call_gpt, call_gemini, call_openrouter
from src.prompts import build_prompt_t1, build_prompt_t2
from src.parsing import extract_address_term, extract_option_number, normalize

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Choose: "gpt", "gemini", or "openrouter"
LLM_PROVIDER = "gemini"
MODEL_NAME = "gemini-2.5-flash" 

CONFIGS_TO_EVALUATE = [
    "address_terms_pronominal",
    "address_terms_nominal"
    # "social_customs"
]

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "results" / "logs" / MODEL_NAME.replace("/", "_")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def call_llm(prompt):
    if LLM_PROVIDER == "gpt":
        return call_gpt(prompt, model=MODEL_NAME)
    elif LLM_PROVIDER == "gemini":
        return call_gemini(prompt, model=MODEL_NAME)
    elif LLM_PROVIDER == "openrouter":
        return call_openrouter(prompt, model=MODEL_NAME)
    else:
        raise ValueError(f"Unknown provider: {LLM_PROVIDER}")

def run_evaluation():
    results = []

    for config in CONFIGS_TO_EVALUATE:
        print(f"\nEvaluating Config: {config} using {MODEL_NAME}")
        
        try:
            dataset = load_dataset("sijantanvir/BanglaSocialBench", config, split="test")
            df = dataset.to_pandas().fillna("")
        except Exception as e:
            print(f"Failed to load HF config {config}: {e}")
            continue

        # Determine task type based on config name
        is_t1 = "pronominal" in config
        correct = 0
        response_logs = []

        for i, row in df.iterrows():
            if is_t1:
                prompt = build_prompt_t1(row)
                raw_out = call_llm(prompt)
                pred = extract_address_term(raw_out)

                golds = {normalize(row.get("Primary", "")), normalize(row.get("Secondary", ""))} - {""}
                is_correct = pred in golds
                gold_out = f"{row.get('Primary', '')} / {row.get('Secondary', '')}".strip(" /")
                parsed_pred = pred
            else:
                prompt = build_prompt_t2(row)
                raw_out = call_llm(prompt)
                pred_num = extract_option_number(raw_out)
                
                # Map extracted number to text
                mapping = {
                    "1": row.get("OptionA", ""), "১": row.get("OptionA", ""),
                    "2": row.get("OptionB", ""), "২": row.get("OptionB", ""),
                    "3": row.get("OptionC", ""), "৩": row.get("OptionC", ""),
                    "4": row.get("OptionD", ""), "৪": row.get("OptionD", ""),
                }
                pred_text = normalize(mapping.get(pred_num, ""))
                gold_text = normalize(row.get("Answer", ""))
                
                is_correct = (pred_text == gold_text)
                gold_out = row.get("Answer", "")
                parsed_pred = pred_text if pred_text else raw_out

            correct += int(is_correct)
            response_logs.append({
                "Situation": row.get("Situation", ""),
                "Question": row.get("Question", ""),
                "LLM_Raw_Response": raw_out,
                "LLM_Parsed_Response": parsed_pred,
                "Gold_Answer": gold_out,
                "Correct": is_correct
            })

            print(f"[{i+1}/{len(df)}] Pred: {parsed_pred} | Gold: {gold_out} | {'✓' if is_correct else '✗'}")

        acc = correct / len(df) if len(df) > 0 else 0
        results.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "LLM": MODEL_NAME,
            "Config": config,
            "Accuracy": round(acc, 4),
            "Correct": correct,
            "Total": len(df)
        })

        # Save responses for this config
        resp_df = pd.DataFrame(response_logs)
        resp_path = OUT_DIR / f"{MODEL_NAME}_{config}_Responses.csv"
        resp_df.to_csv(resp_path, index=False)
        print(f"Saved logs to {resp_path}")

    # Update summary
    summary_df = pd.DataFrame(results)
    summary_path = OUT_DIR / "summary_results.csv"
    if summary_path.exists():
        summary_df = pd.concat([pd.read_csv(summary_path), summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print("\n=== FINAL SUMMARY ===")
    print(summary_df)

if __name__ == "__main__":
    run_evaluation()
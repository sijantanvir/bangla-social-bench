import pandas as pd
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from src.parsing import normalize, extract_choice_letter

# Depending on the model, import the client that supports logprobs
# from src.llm_clients import call_together as call_logprob_llm
from src.llm_clients import call_gpt

MODEL_NAME = "gpt-4o-mini"
CONFIG = "address_terms_pronominal"

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "results" / "logs" / f"{MODEL_NAME}_logprobs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_logprob_prompt(row):
    situation = row.get("Situation", "").strip()
    return f"""প্রেক্ষাপট:
{situation}
প্রশ্ন:
{row['Question']}
উত্তর নির্বাচন করুন (শুধু একটি অক্ষর লিখুন):
A. আপনি
B. তুমি
C. তুই
Output Format: A/B/C
""".strip()

def serialize_topk_logprobs(logprobs, k=5):
    """
    Adjust this based on the client (OpenAI vs Together).
    Currently implemented for OpenAI structure.
    """
    if logprobs is None:
        return ""
    try:
        # For OpenAI:
        top = logprobs["content"][0]["top_logprobs"][:k]
        parts = [f"{t['token'].replace(chr(10), '\\n')}:{round(t['logprob'], 4)}" for t in top]
        return ", ".join(parts)
    except Exception:
        return ""

def run_logprob_collection():
    print(f"Collecting logprobs for {CONFIG} using {MODEL_NAME}")
    
    dataset = load_dataset("sijantanvir/BanglaSocialBench", CONFIG, split="test")
    df = dataset.to_pandas().fillna("")
    
    logs = []
    
    for idx, row in df.iterrows():
        prompt = build_logprob_prompt(row)
        
        # Modify the underlying client call if you need to pass logprobs=True
        # For this script to work out-of-the-box, ensure your src.llm_clients.call_gpt 
        # returns (text, logprobs) when logprobs=True is uncommented.
        raw_out, logprobs = call_gpt(prompt, model=MODEL_NAME) # Assumes modified client
        
        pred_letter = extract_choice_letter(raw_out)
        pred_term = {"A": "আপনি", "B": "তুমি", "C": "তুই"}.get(pred_letter, "")
        
        golds = {normalize(row.get("Primary", "")), normalize(row.get("Secondary", ""))} - {""}
        is_correct = normalize(pred_term) in golds
        
        logs.append({
            "Situation": row.get("Situation", ""),
            "Question": row.get("Question", ""),
            "LLM_Raw_Response": raw_out,
            "LLM_Parsed_Response": pred_term,
            "Correct": is_correct,
            "TopK_LogProbs": serialize_topk_logprobs(logprobs)
        })
        print(f"[{idx+1}/{len(df)}] Pred={pred_letter} | Correct={is_correct}")

    out_csv = OUT_DIR / f"{MODEL_NAME}_{CONFIG}_Responses.csv"
    pd.DataFrame(logs).to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

if __name__ == "__main__":
    run_logprob_collection()
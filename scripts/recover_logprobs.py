import math
import pandas as pd
from pathlib import Path

MODEL_NAME = "gpt-4o-mini_logprobs"
CONFIG = "address_terms_pronominal"

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "results" / "logs" / MODEL_NAME

VALID_LETTERS = ["A", "B", "C"]

def parse_logprob_string(s):
    out = {}
    if not isinstance(s, str) or not s.strip():
        return out
        
    for part in s.split(","):
        if ":" not in part:
            continue
        tok, lp = part.split(":", 1)
        tok = tok.strip().replace("Ġ", "")
        if tok not in VALID_LETTERS:
            continue
        try:
            lp = float(lp.strip())
        except ValueError:
            continue
        if tok not in out:
            out[tok] = lp
        if len(out) == len(VALID_LETTERS):
            break
    return out

def compute_choice_probs(logprob_dict):
    scores = {}
    for k in VALID_LETTERS:
        if k in logprob_dict:
            scores[k] = math.exp(logprob_dict[k])
    
    if not scores:
        return {k: 0.0 for k in VALID_LETTERS}
        
    Z = sum(scores.values())
    return {k: (scores.get(k, 0.0) / Z) for k in VALID_LETTERS}

def recover_logprobs():
    csv_path = LOG_DIR / f"{MODEL_NAME.replace('_logprobs', '')}_{CONFIG}_Responses.csv"
    
    if not csv_path.exists():
        print(f"⚠️ Missing CSV: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    for idx, row in df.iterrows():
        logprob_dict = parse_logprob_string(row.get("TopK_LogProbs", ""))
        probs = compute_choice_probs(logprob_dict)
        
        df.at[idx, "Prob_A (আপনি)"] = probs["A"]
        df.at[idx, "Prob_B (তুমি)"] = probs["B"]
        df.at[idx, "Prob_C (তুই)"] = probs["C"]

    df.to_csv(csv_path, index=False)
    print(f"✅ Recovered probabilities saved to: {csv_path.name}")

if __name__ == "__main__":
    recover_logprobs()
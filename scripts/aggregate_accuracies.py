import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "results" / "logs"
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def compute_micro_accuracy(csv_paths):
    correct, total = 0, 0
    for p in csv_paths:
        df = pd.read_csv(p)
        if "Correct" in df.columns:
            correct += df["Correct"].astype(int).sum()
            total += len(df)
    return (correct / total, correct, total) if total > 0 else (None, 0, 0)

def run_aggregation():
    rows = []
    for llm_dir in LOG_DIR.iterdir():
        if not llm_dir.is_dir(): continue
        
        pron_csvs = list(llm_dir.glob("*_address_terms_pronominal_Responses.csv"))
        nom_csvs = list(llm_dir.glob("*_address_terms_nominal_Responses.csv"))
        kin_csvs = list(llm_dir.glob("*_kinship_reasoning_Responses.csv"))
        cust_csvs = list(llm_dir.glob("*_social_customs_Responses.csv"))
        
        acc_pron, pron_c, pron_t = compute_micro_accuracy(pron_csvs)
        acc_nom, nom_c, nom_t = compute_micro_accuracy(nom_csvs)
        acc_kin, _, kin_t = compute_micro_accuracy(kin_csvs)
        acc_cust, _, cust_t = compute_micro_accuracy(cust_csvs)
        
        addr_c = pron_c + nom_c
        addr_t = pron_t + nom_t
        addr_acc = addr_c / addr_t if addr_t > 0 else None
        
        accs = [a for a in [addr_acc, acc_kin, acc_cust] if a is not None]
        macro_acc = sum(accs) / len(accs) if accs else None
        
        if macro_acc:
            rows.append({
                "LLM": llm_dir.name,
                "Address_Term_Accuracy": addr_acc,
                "Kinship_Accuracy": acc_kin,
                "Social_Customs_Accuracy": acc_cust,
                "Macro_Cultural_Accuracy": macro_acc
            })

    if not rows:
        print("No valid logs found to aggregate.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "Overall_Cultural_Accuracy.csv", index=False)
    
    # -- Plotting --
    df["Accuracy"] = df["Macro_Cultural_Accuracy"] * 100
    df["Family"] = df["LLM"].apply(lambda x: "GPT" if "gpt" in x.lower() else "Gemini" if "gemini" in x.lower() else "Claude" if "claude" in x.lower() else "LLaMA" if "llama" in x.lower() else "Gemma" if "gemma" in x.lower() else "Qwen" if "qwen" in x.lower() else "DeepSeek" if "deepseek" in x.lower() else "Other")
    
    df = df.sort_values(by=["Family", "Accuracy"]).reset_index(drop=True)
    
    plt.figure(figsize=(14, 4))
    plt.bar(np.arange(len(df)), df["Accuracy"], width=0.72, color="steelblue")
    plt.axhline(93.69, linestyle="--", color="red", linewidth=1.4)
    plt.text(len(df)-0.5, 93.69-8, "Human Baseline", color="red", ha="right", fontweight="bold")
    
    plt.ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    plt.xticks(np.arange(len(df)), df["LLM"], rotation=35, ha="right")
    plt.ylim(0, 100)
    plt.title("Overall Cultural Accuracy", fontsize=14, pad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "overall_cultural_accuracy.pdf", dpi=300, bbox_inches="tight")
    print(f"✅ Saved CSV and Plot to {FIG_DIR}")

if __name__ == "__main__":
    run_aggregation()
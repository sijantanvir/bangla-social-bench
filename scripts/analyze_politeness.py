import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from pathlib import Path
from datasets import load_dataset
from src.parsing import extract_address_term, normalize

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "results" / "logs"
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

POLITENESS_RANK = {"tui": 1, "tumi": 2, "apni": 3}

def extract_pronoun(text):
    if not isinstance(text, str): return None
    if "আপনি" in text: return "apni"
    if "তুমি" in text: return "tumi"
    if "তুই" in text: return "tui"
    return None

def analyze_politeness():
    # Load gold annotations from HF
    dataset = load_dataset("sijantanvir/BanglaSocialBench", "address_terms_pronominal", split="test")
    gold_df = dataset.to_pandas().fillna("")
    
    gold_lookup = {}
    for _, row in gold_df.iterrows():
        key = (normalize(row.get("Situation", "")), normalize(row.get("Question", "")))
        gold_lookup[key] = {
            "Primary": extract_pronoun(row.get("Primary", "")),
            "Secondary": extract_pronoun(row.get("Secondary", ""))
        }

    records = []
    
    # Iterate dynamically through all model folders in results/logs/
    for model_dir in LOG_DIR.iterdir():
        if not model_dir.is_dir(): continue
        llm_name = model_dir.name
        
        # Look for pronominal response files
        for csv_path in model_dir.glob("*_address_terms_pronominal_Responses.csv"):
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                pred = extract_pronoun(row.get("LLM_Parsed_Response", ""))
                if pred is None: continue
                
                key = (normalize(row.get("Situation", "")), normalize(row.get("Question", "")))
                gold = gold_lookup.get(key)
                if not gold: continue
                
                gold_set = {gold["Primary"], gold["Secondary"]} - {None}
                if not gold_set: continue
                
                gold_ranks = [POLITENESS_RANK[g] for g in gold_set]
                pred_rank = POLITENESS_RANK[pred]
                
                if pred in gold_set:
                    label = "Correct"
                elif pred_rank > max(gold_ranks):
                    label = "Over-Polite"
                elif pred_rank < min(gold_ranks):
                    label = "Under-Polite"
                else:
                    label = "Correct"
                    
                records.append({"LLM": llm_name, "Label": label})

    # Aggregation & Stats
    res_df = pd.DataFrame(records)
    if res_df.empty:
        print("No matching logs found to analyze.")
        return

    counts = res_df.groupby(["LLM", "Label"]).size().reset_index(name="Count")
    totals = counts.groupby("LLM")["Count"].sum().reset_index(name="Total")
    pct = counts.merge(totals, on="LLM")
    pct["Percent"] = 100 * pct["Count"] / pct["Total"]
    
    pivot = pct.pivot(index="LLM", columns="Label", values="Percent").fillna(0)
    for col in ["Under-Polite", "Correct", "Over-Polite"]:
        if col not in pivot: pivot[col] = 0
    
    pivot = pivot[["Under-Polite", "Correct", "Over-Polite"]]
    print("\nDirectional politeness distribution (%):")
    print(pivot.round(2))

    # Plotting
    models = pivot.index.tolist()
    under_vals = -pivot["Under-Polite"].values
    over_vals = pivot["Over-Polite"].values
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(y, under_vals, color="#2E6E91", label="Under-politeness", height=0.60, zorder=3)
    ax.barh(y, over_vals, color="#D0901E", label="Over-politeness", height=0.60, zorder=3)
    ax.axvline(0, color="black", linewidth=0.9, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Inappropriate Use Rate (%)", fontsize=12)
    ax.set_title("Asymmetry in Politeness Deviations", fontsize=14, pad=8)
    
    max_val = max(over_vals.max() if len(over_vals) else 0, abs(under_vals).max() if len(under_vals) else 0)
    ax.set_xlim(-max_val * 1.05, max_val * 1.05)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    plt.tight_layout()
    plot_path = FIG_DIR / "over_under_polite.pdf"
    plt.savefig(plot_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    analyze_politeness()
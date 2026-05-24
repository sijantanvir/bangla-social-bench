import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from datasets import load_dataset

# ===============================
# Helpers
# ===============================
def norm(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x if x else None

def is_multilabel_sheet(df):
    has_primary = any(col.startswith("Primary_") for col in df.columns)
    has_answer = any(col.startswith("Answer_") for col in df.columns)
    return has_primary and not has_answer

def get_collapsed_label(row, idx):
    if f"Primary_{idx}" in row.index:
        p = norm(row.get(f"Primary_{idx}"))
        if p is not None: return p
        
        s = norm(row.get(f"Secondary_{idx}"))
        if s is not None: return s

    a = norm(row.get(f"Answer_{idx}"))
    if a is not None: return a
    return None

def get_label_set(row, idx):
    labels = set()
    p = norm(row.get(f"Primary_{idx}"))
    s = norm(row.get(f"Secondary_{idx}"))
    if p: labels.add(p)
    if s: labels.add(s)
    return labels

# ===============================
# Main Execution
# ===============================
def run_iaa():
    CONFIGS = [
        "address_terms_pronominal",
        "address_terms_nominal",
        "kinship_reasoning",
        "social_customs"
    ]
    
    standard_results = []
    binary_results = []

    all_std_r1 = []
    all_std_r2 = []
    global_binary_kappas = []

    print("\n=== Computing Inter-Annotator Agreement ===\n")

    for config in CONFIGS:
        try:
            df = load_dataset("sijantanvir/BanglaSocialBench", config, split="test").to_pandas()
        except Exception as e:
            print(f"Skipping {config}: {e}")
            continue
            
        # Standardize columns if dataset uses 'Primary_1' and 'Primary_2'
        # If it just uses 'Primary' and 'Secondary' as annotators, adjust indices below.
        
        multilabel = is_multilabel_sheet(df)

        # ---------- STANDARD κ ----------
        r1 = df.apply(lambda r: get_collapsed_label(r, 1), axis=1)
        r2 = df.apply(lambda r: get_collapsed_label(r, 2), axis=1)

        mask = r1.notna() & r2.notna()
        r1 = r1[mask]
        r2 = r2[mask]

        n_items = len(r1)
        matches = int((r1 == r2).sum())
        agreement_rate = matches / n_items if n_items > 0 else np.nan

        if n_items >= 2:
            kappa_std = cohen_kappa_score(r1, r2)
            all_std_r1.extend(r1.tolist())
            all_std_r2.extend(r2.tolist())
        else:
            kappa_std = np.nan

        standard_results.append({
            "Config": config,
            "N_Items": n_items,
            "Matches": matches,
            "Agreement_Rate": round(agreement_rate, 4),
            "Standard_Kappa": round(kappa_std, 4),
        })

        # ---------- BINARY κ (multi-label only) ----------
        if multilabel:
            r1_sets = df.apply(lambda r: get_label_set(r, 1), axis=1).tolist()
            r2_sets = df.apply(lambda r: get_label_set(r, 2), axis=1).tolist()

            label_universe = sorted(set().union(*r1_sets, *r2_sets))
            label_kappas = []

            for label in label_universe:
                v1 = [1 if label in s else 0 for s in r1_sets]
                v2 = [1 if label in s else 0 for s in r2_sets]

                k = cohen_kappa_score(v1, v2, labels=[0, 1])

                if np.isnan(k):
                    agreement = np.mean(np.array(v1) == np.array(v2))
                    k = 1.0 if agreement == 1.0 else 0.0

                label_kappas.append(k)
                global_binary_kappas.append(k)

            binary_results.append({
                "Config": config,
                "N_Items": len(df),
                "Unique_Labels": len(label_universe),
                "Binary_Kappa_Mean": round(np.mean(label_kappas), 4),
            })

            print(f"{config:30s} | Matches = {matches:4d}/{n_items} | κ = {kappa_std:.4f} | Binary κ = {np.mean(label_kappas):.4f}")
        else:
            print(f"{config:30s} | Matches = {matches:4d}/{n_items} | κ = {kappa_std:.4f} | Binary κ = excluded (single-answer)")

    # ===============================
    # GLOBAL METRICS
    # ===============================
    global_standard_kappa = cohen_kappa_score(all_std_r1, all_std_r2) if len(all_std_r1) >= 2 else np.nan
    global_match_rate = np.mean([r1 == r2 for r1, r2 in zip(all_std_r1, all_std_r2)]) if all_std_r1 else np.nan
    global_binary_kappa = np.mean(global_binary_kappas) if global_binary_kappas else np.nan

    print("\n=== PER-CONFIG STANDARD κ ===")
    print(pd.DataFrame(standard_results))

    if binary_results:
        print("\n=== PER-CONFIG BINARY κ (MULTI-LABEL ONLY) ===")
        print(pd.DataFrame(binary_results))

    print("\n=== GLOBAL AGREEMENT ===")
    print(f"Global Standard κ: {global_standard_kappa:.4f}")
    print(f"Global Raw Agreement: {global_match_rate:.4f}")
    
    if not np.isnan(global_binary_kappa):
        print(f"Global Binary κ (macro over labels): {global_binary_kappa:.4f}")

if __name__ == "__main__":
    run_iaa()
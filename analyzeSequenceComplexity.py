from datasets import load_dataset
import pandas as pd
import re

# 1. Dataseti yükle
ds = load_dataset("ibm-research/MermaidSeqBench")
df = pd.DataFrame(ds['train'])

def analyze_sequence_metrics(mermaid_code):
    """
    IJSEA (2023) standartlarına göre NOL, MIL, MOL ve C_system hesaplar.
    """
    if not isinstance(mermaid_code, str) or "sequenceDiagram" not in mermaid_code:
        return 0, 0, 0, 0

    # Mesaj akışlarını yakala (Sender ->> Receiver vb.)
    flows = re.findall(r"(\w+)\s*(?:->>|->|-->>|--)\s*(\w+)", mermaid_code)
    
    # Katılımcıları (Lifelines) belirle
    explicit_p = re.findall(r"(?:participant|actor)\s+(\w+)", mermaid_code)
    lifelines = set(explicit_p)
    for s, r in flows:
        lifelines.update([s, r])
    
    nol = len(lifelines)
    if nol == 0: return 0, 0, 0, 0

    total_mil = 0
    total_mol = 0
    c_i_sum = 0
    
    # Her lifeline için Ci = (MIL * MOL)^2
    for actor in lifelines:
        mil_actor = sum(1 for _, r in flows if r == actor)
        mol_actor = sum(1 for s, _ in flows if s == actor)
        
        total_mil += mil_actor
        total_mol += mol_actor
        c_i_sum += (mil_actor * mol_actor) ** 2
    
    # C_system = Sum(Ci) * NOL
    c_system = c_i_sum * nol
    return nol, total_mil, total_mol, c_system

# 2. Metrikleri hesapla ve yeni sütunlara ata
metrics = df['expected_output'].apply(analyze_sequence_metrics)
df[['NOL', 'Total_MIL', 'Total_MOL', 'C_system']] = pd.DataFrame(metrics.tolist(), index=df.index)

# 3. Complexity Level (Eşik değerlerini IBM datasetine göre belirleyelim)
# Not: IBM verisi genelde daha kompleks olduğu için 150 ve 1500 gibi sınırları deneyebilirsin
def categorize(score):
    if score <= 150: return 'Low'
    if score <= 1500: return 'Medium'
    return 'High'

df['Complexity_Level'] = df['C_system'].apply(categorize)

# 4. İlk 5 sonucu ve dağılımı gör
#print(df[['nl_task_title', 'NOL', 'Total_MIL', 'Total_MOL', 'C_system', 'Complexity_Level']].head())
pd.set_option('display.max_rows', None) # Tüm satırları göster
print(df[['nl_task_title', 'NOL', 'Total_MIL', 'Total_MOL', 'C_system', 'Complexity_Level']])
print("\nDağılım:\n", df['Complexity_Level'].value_counts())
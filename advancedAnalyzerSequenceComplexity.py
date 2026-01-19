from datasets import load_dataset
import pandas as pd
import re

# 1. Dataseti yükle
ds = load_dataset("ibm-research/MermaidSeqBench")
df = pd.DataFrame(ds['train'])

def analyze_advanced_sequence_metrics(mermaid_code):
    """
    UML 2.x ve IJSEA (2023) tabanlı geliştirilmiş karmaşıklık modeli.
    Formül: [Sum(Ci) + (W_cf * Total_CF) + (W_io * Total_IO)] * NOL
    """
    if not isinstance(mermaid_code, str) or "sequenceDiagram" not in mermaid_code:
        return 0, 0, 0, 0, 0, 0

    # --- Temel Yapı Analizi (MIL, MOL, NOL) ---
    flows = re.findall(r"(\w+)\s*(?:->>|->|-->>|--)\s*(\w+)", mermaid_code)
    explicit_p = re.findall(r"(?:participant|actor)\s+(\w+)", mermaid_code)
    lifelines = set(explicit_p)
    for s, r in flows:
        lifelines.update([s, r])
    
    nol = len(lifelines)
    if nol == 0: return 0, 0, 0, 0, 0, 0

    c_i_sum = 0
    total_mil = 0
    total_mol = 0
    for actor in lifelines:
        mil_actor = sum(1 for _, r in flows if r == actor)
        mol_actor = sum(1 for s, _ in flows if s == actor)
        total_mil += mil_actor
        total_mol += mol_actor
        c_i_sum += (mil_actor * mol_actor) ** 2

    # --- Mantıksal Yapı Analizi (CF, IO) ---
    # Total CF: alt, loop, opt, par, break, rect sayımı
    cf_patterns = r"\b(alt|loop|opt|par|break|rect)\b"
    cf_list = re.findall(cf_patterns, mermaid_code)
    total_cf = len(cf_list)

    # Total IO: Her CF bir operand ile başlar, her 'else' veya 'and' ek bir operanddır
    else_and_list = re.findall(r"\b(else|and)\b", mermaid_code)
    total_io = total_cf + len(else_and_list)

    # --- Ağırlıklandırma ve Final Hesaplama ---
    w_cf = 5  # Combined Fragments ağırlığı
    w_io = 2  # Interaction Operands ağırlığı
    
    # Yeni Formül Uygulaması
    c_system = (c_i_sum + (w_cf * total_cf) + (w_io * total_io)) * nol
    
    return nol, total_mil, total_mol, total_cf, total_io, c_system

# 2. Metrikleri hesapla ve yeni sütunlara ata
metrics_results = df['expected_output'].apply(analyze_advanced_sequence_metrics)
df[['NOL', 'Total_MIL', 'Total_MOL', 'Total_CF', 'Total_IO', 'C_system']] = pd.DataFrame(metrics_results.tolist(), index=df.index)

# 3. İstatistiksel Eşikleme (Statistical Thresholding)
# IBM dataseti çok yoğun olduğu için sabit sayı yerine çeyrekliklere (quartiles) göre bölüyoruz
q1 = df['C_system'].quantile(0.25)
q3 = df['C_system'].quantile(0.75)

def categorize_dynamic(score):
    if score <= q1: return 'Low'
    if score <= q3: return 'Medium'
    return 'High'

df['Complexity_Level'] = df['C_system'].apply(categorize_dynamic)

# 4. Sonuçları Görüntüle
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(f"--- İstatistiksel Eşikler ---")
print(f"Low: 0 - {q1:.2f} | Medium: {q1:.2f} - {q3:.2f} | High: > {q3:.2f}\n")

print(df[['nl_task_title', 'NOL', 'Total_MIL', 'Total_CF', 'Total_IO', 'C_system', 'Complexity_Level']])
print("\nKarmaşıklık Seviyesi Dağılımı:\n", df['Complexity_Level'].value_counts())
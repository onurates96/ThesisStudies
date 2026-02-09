import pandas as pd
import re
from datasets import load_dataset

# 1. Load the Dataset
print("Loading dataset: Celiadraw/text-to-mermaid...")
ds = load_dataset("Celiadraw/text-to-mermaid")
df = pd.DataFrame(ds['train'])

# 2. Diagram Type Identification Logic
def identify_type(code):
    code = str(code).lower()
    if "classdiagram" in code: return "Class"
    if "sequencediagram" in code: return "Sequence"
    if "statediagram" in code: return "State"
    if "graph td" in code or "graph lr" in code or "flowchart" in code: return "Action/Flow"
    return "Other"

df['diagram_type'] = df['output'].apply(identify_type)

# --- 3. ACADEMIC COMPLEXITY FUNCTIONS ---

def analyze_sequence(code):
    """
    Section 2.8: Sequence Diagram Complexity
    Formula: C_system = [Sum(Ci) + (W_cf * Total_CF) + (W_io * Total_IO)] * NOL
    """
    # Identify message flows and participants (NOL)
    flows = re.findall(r"(\w+)\s*(?:->>|->|-->>|--)\s*(\w+)", code)
    lifelines = set(re.findall(r"(?:participant|actor)\s+(\w+)", code))
    for s, r in flows: lifelines.update([s, r])
    nol = len(lifelines)
    if nol == 0: return 0

    # Ci = (MIL * MOL)^2 (Interaction Density per Lifeline)
    c_i_sum = 0
    for actor in lifelines:
        mil = sum(1 for _, r in flows if r == actor)
        mol = sum(1 for s, _ in flows if s == actor)
        c_i_sum += (mil * mol) ** 2

    # Combined Fragments (CF) and Interaction Operands (IO)
    cf_list = re.findall(r"\b(alt|loop|opt|par|break|rect)\b", code)
    total_cf = len(cf_list)
    total_io = total_cf + len(re.findall(r"\b(else|and)\b", code))
    
    # Calculation with weights: W_cf=5, W_io=2
    return (c_i_sum + (5 * total_cf) + (2 * total_io)) * nol

def analyze_class(code):
    """
    Section 2.9: Class Diagram Complexity
    Formula: C_system = [Sum(Ci) + R_total] * NC
    """
    classes = set(re.findall(r"class\s+(\w+)", code))
    nc = len(classes)
    if nc == 0: return 0
    
    # Element Load (Attributes Ai, Methods Mi)
    # Weights: wA=1, wM=2
    attributes = len(re.findall(r"[\+\-\#]\w+(?!\()", code))
    methods = len(re.findall(r"[\+\-\#]\w+\(.*\)", code))
    el_total = (attributes * 1) + (methods * 2) 
    
    # R_total (Relationship weights H_type and Multiplicity W_mult)
    # Weights from Manso (2003)
    weights = {'--|>': 5, '..|>': 5, '*--': 4, 'o--': 3, '-->': 2, '..>': 1}
    r_total = 0
    for rel, w in weights.items():
        count = code.count(rel)
        # Apply multiplicity factor (W_mult = 1.1 as a structural average)
        r_total += (count * w * 1.1) 
        
    return (el_total + r_total) * nc

def analyze_action_flow(code):
    """
    Section 2.10: Activity Diagram Complexity
    Formula: C_system = [Sum(Ci) + C_cf] * NN
    """
    # NN (Total Nodes) and Control Flow Edges
    nodes = re.findall(r"\[.+?\]|\(\(.+?\)\)|\{.+\}", code)
    nn = len(nodes)
    edges = len(re.findall(r"-->", code))
    if nn == 0: return 0
    
    # C_cf (Control Flow Complexity: Decisions, Parallelism, Guards)
    nd = len(re.findall(r"\{.*?\}", code)) # Decision Nodes
    np = len(re.findall(r"fork|join", code, re.I)) # Parallel Constructs
    ng = len(re.findall(r"\|.+?\|", code)) # Guards
    c_cf = (3 * nd) + (4 * np) + (2 * ng)
    
    # Interaction Density Estimation: (EIN * EOUT)^2
    avg_interaction = ((edges/nn) ** 2) if nn > 0 else 0
    
    return ( (nn * avg_interaction) + c_cf ) * nn

def analyze_state(code):
    """
    Section 2.11: Statechart Complexity (CSC Score)
    Formula: CSC = CFF + (gamma * EEA) + (delta * NA)
    """
    # WNS (Weighted Number of States)
    simple_states = len(re.findall(r"state\s+(\w+)", code))
    composite_states = len(re.findall(r"state\s+\w+\s*\{", code))
    wns = (simple_states * 1) + (composite_states * 3)
    
    # CFF (Control Flow Features: Transitions + Guards)
    transitions = code.count("-->")
    guards = code.count("[")
    cff = wns + transitions + guards
    
    # EEA (Entry/Exit Actions) and NA (Activities)
    eea = len(re.findall(r"entry/|exit/", code, re.I))
    na = len(re.findall(r"do/", code, re.I))
    
    # CSC = CFF + EEA + NA (Assuming gamma=delta=1)
    return cff + eea + na

# --- 4. DATA PROCESSING PIPELINE ---

def calculate_master_complexity(row):
    dtype = row['diagram_type']
    code = row['output']
    if dtype == "Sequence": return analyze_sequence(code)
    if dtype == "Class": return analyze_class(code)
    if dtype == "Action/Flow": return analyze_action_flow(code)
    if dtype == "State": return analyze_state(code)
    return 0

# Apply the complexity metrics
df['C_system'] = df.apply(calculate_master_complexity, axis=1)

# Categorize and Filter Subsets
subsets = {
    "Class": df[df['diagram_type'] == "Class"].copy(),
    "Sequence": df[df['diagram_type'] == "Sequence"].copy(),
    "State": df[df['diagram_type'] == "State"].copy(),
    "Action/Flow": df[df['diagram_type'] == "Action/Flow"].sample(n=min(750, len(df[df['diagram_type'] == "Action/Flow"])), random_state=42).copy()
}

# 5. Statistical Thresholding and Reporting
for name, subset in subsets.items():
    if not subset.empty:
        # Determine thresholds based on dataset quartiles
        q1 = subset['C_system'].quantile(0.25)
        q3 = subset['C_system'].quantile(0.75)
        
        subset['Complexity_Level'] = subset['C_system'].apply(
            lambda s: 'Low' if s <= q1 else ('Medium' if s <= q3 else 'High')
        )
        
        print(f"\n=== {name.upper()} DIAGRAM ANALYSIS ===")
        print(f"Total Samples: {len(subset)}")
        print(f"Thresholds: Low <= {q1:.2f} < Medium <= {q3:.2f} < High")
        print("Distribution:")
        print(subset['Complexity_Level'].value_counts())
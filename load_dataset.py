import ollama
from datasets import load_dataset
import pandas as pd
import json
import re
from datetime import datetime
from run_mermaid_benchmark import run_mermaid_benchmark
from get_custom_prompt import get_custom_prompt

# 1. Load the dataset
ds = load_dataset("Celiadraw/text-to-mermaid")
#test_samples = ds['train'].select(range(20)) # Start with 20 samples for initial testing

# Sütun isimlerini görelim
print("Column Names:", ds['train'].column_names)

df = pd.DataFrame(ds['train'])
print("\nVeri Seti Önizleme:")
print(df[['prompt', 'output']])

# 2. Categorize based on the 'output' content
def identify_type(code):
    code = str(code).lower()
    if "classdiagram" in code: return "Class"
    if "sequencediagram" in code: return "Sequence"
    if "statediagram" in code: return "State"
    if "graph td" in code or "graph lr" in code: return "Component/Flow"
    return "Other"

df['diagram_type'] = df['output'].apply(identify_type)

# 3. View the distribution
print("--- Dataset Categorization Statistics ---")
print(df['diagram_type'].value_counts())

# 4. Filter a specific subset for testing (e.g., Class Diagrams)
class_subset = df[df['diagram_type'] == "Class"].copy()
sequence_subset = df[df['diagram_type'] == "Sequence"].copy()
state_subset = df[df['diagram_type'] == "State"].copy()
action_subset_all = df[df['diagram_type'] == "Component/Flow"].copy()
action_subset = action_subset_all.sample(n=750, random_state=42)

all_subsets = {
    "Class": class_subset,
    "Sequence": sequence_subset,
    "State": state_subset,
    "Action/Flow": action_subset
}


output_file = f"benchmark_raw_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

# Models (okey for 32GB RAM)
test_models = [
    "llama3.1:8b", 
    "deepseek-coder-v2:16b", 
    "qwen2.5-coder:14b", 
    "gemma2:9b",
    "phi3.5:latest"  # Microsoft'un verimli modeli
]

# llama3.1:8b 4.9 GB
# deepseek-coder-v2:16b 8.9 GB
# qwen2.5-coder:14b 9.0 GB
# gemma2:9b 5.4 GB
# phi3.5:latest 2.2 GB

SAMPLE_SIZE_PER_CATEGORY = 10

subsets_to_test = {
        "Class": class_subset,
        "Sequence": sequence_subset,
        "State": state_subset,
        "Action/Flow": action_subset
    }

all_benchmark_data = []

print("--- Starting Mermaid Benchmarking ---")

# 3. Execution Loop
for category_name, dataframe in subsets_to_test.items():
    # Call the modular function
    category_results = run_mermaid_benchmark(
        subset_df=dataframe,
        subset_name=category_name,
        models=test_models,
        prompt_func=get_custom_prompt,
        sample_size=SAMPLE_SIZE_PER_CATEGORY
    )
    all_benchmark_data.append(category_results)

# 4. Consolidate and Export
final_report = pd.concat(all_benchmark_data, ignore_index=True)
    
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_filename = f"mermaid_bench_results_{timestamp}.csv"
    
final_report.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
print("-" * 30)
print(f"✅ BENCHMARK COMPLETE")
print(f"File Saved: {output_filename}")
print("-" * 30)

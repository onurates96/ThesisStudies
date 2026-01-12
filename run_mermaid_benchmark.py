import ollama
import pandas as pd
from datetime import datetime

def run_mermaid_benchmark(subset_df, subset_name, models, prompt_func, sample_size=5):
    """
    Tests selected models on a specific subset and returns the benchmarking results.
    
    Args:
        subset_df (pd.DataFrame): The dataframe containing the test cases.
        subset_name (str): Name of the UML category (e.g., 'Sequence', 'Class').
        models (list): List of Ollama model names to be tested.
        prompt_func (function): The external function used to generate expert prompts.
        sample_size (int): Number of samples to process from the subset.
        
    Returns:
        pd.DataFrame: A collection of prompts, ground truths, and model-generated outputs.
    """
    results = []
    # Select the top N samples from the subset
    test_samples = subset_df.head(sample_size)
    
    print(f"\n🚀 Starting Benchmark for Category: {subset_name} (Samples: {sample_size})")
    
    for idx, row in test_samples.iterrows():
        story = row['prompt']
        ground_truth = row['output']
        
        # Generate the expert prompt using the external library
        expert_prompt = prompt_func(story, subset_name)
        
        entry = {
            "category": subset_name,
            "original_id": idx,
            "input_story": story,
            "ground_truth": ground_truth,
            "generated_prompt": expert_prompt
        }
        
        for model in models:
            print(f"   - Processing Model: {model} (Sample ID: {idx})")
            try:
                # Local inference via Ollama
                response = ollama.chat(
                    model=model, 
                    messages=[{'role': 'user', 'content': expert_prompt}]
                )
                
                # Sanitize column name for CSV compatibility
                model_col_name = f"actual_{model.replace(':', '_').replace('.', '_')}"
                entry[model_col_name] = response['message']['content']
                
            except Exception as e:
                # Log error in the dataframe instead of crashing the script
                entry[f"actual_{model}"] = f"Error: {str(e)}"
        
        results.append(entry)
        
    return pd.DataFrame(results)

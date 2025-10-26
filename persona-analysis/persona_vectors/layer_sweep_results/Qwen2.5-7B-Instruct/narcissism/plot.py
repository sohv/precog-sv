import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_sweep_log(file_path):
    """
    Parses the layer sweep log file to extract narcissism and coherence scores for each layer.
    
    Args:
        file_path (str): The path to the log file.
        
    Returns:
        pd.DataFrame: A DataFrame with columns ['layer', 'narcissism', 'narcissism_std', 'coherence', 'coherence_std'].
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find blocks of data for each layer
    # It captures the layer number, narcissism mean/std, and coherence mean/std
    pattern = re.compile(
        r"--- Testing Layer (\d+) ---.*?"
        r"narcissism:\s+([\d.]+)\s+\+-\s+([\d.]+)\n"
        r"coherence:\s+([\d.]+)\s+\+-\s+([\d.]+)",
        re.DOTALL  # . matches newline
    )

    matches = pattern.findall(content)
    
    data = []
    for match in matches:
        data.append({
            'layer': int(match[0]),
            'narcissism': float(match[1]),
            'narcissism_std': float(match[2]),
            'coherence': float(match[3]),
            'coherence_std': float(match[4]),
        })
        
    if not data:
        raise ValueError("No data could be parsed from the log file. Check the file format.")
        
    df = pd.DataFrame(data)
    df = df.sort_values(by='layer').reset_index(drop=True)
    
    return df

def plot_layer_sweep(df, trait_name, output_dir="analysis_results"):
    """
    Generates and saves a plot of trait score and coherence vs. layer.

    Args:
        df (pd.DataFrame): DataFrame containing the parsed data.
        trait_name (str): The name of the trait being plotted (e.g., 'narcissism').
        output_dir (str): The directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Narcissism score with its error band
    ax.plot(df['layer'], df[trait_name], marker='o', linestyle='-', label=f'{trait_name.capitalize()} Score')
    ax.fill_between(
        df['layer'],
        df[trait_name] - df[f'{trait_name}_std'],
        df[trait_name] + df[f'{trait_name}_std'],
        alpha=0.2,
        label=f'{trait_name.capitalize()} Std. Dev.'
    )

    # Plot Coherence score with its error band
    ax.plot(df['layer'], df['coherence'], marker='o', linestyle='--', label='Coherence Score')
    ax.fill_between(
        df['layer'],
        df['coherence'] - df['coherence_std'],
        df['coherence'] + df['coherence_std'],
        alpha=0.2,
        label='Coherence Std. Dev.'
    )
    
    # Highlight the "sweet spot" and the coherence drop-off
    peak_narcissism_layer = df.loc[df[trait_name].idxmax()]['layer']
    ax.axvline(x=peak_narcissism_layer, color='r', linestyle='--', linewidth=1.5, label=f'Peak {trait_name.capitalize()} Effect (Layer {peak_narcissism_layer})')
    
    # You can also mark where coherence starts to drop significantly
    # For example, let's mark where coherence drops below 80
    coherence_drop_layer = df[df['coherence'] < 80]['layer'].min()
    if pd.notna(coherence_drop_layer):
        ax.axvline(x=coherence_drop_layer, color='g', linestyle=':', linewidth=1.5, label=f'Coherence Drop-off (Layer {coherence_drop_layer})')

    # Formatting the plot
    ax.set_title(f'Steering Effectiveness vs. Coherence for "{trait_name.capitalize()}" across Layers')
    ax.set_xlabel('Model Layer')
    ax.set_ylabel('Score (0-100)')
    ax.legend()
    ax.set_ylim(0, 105) # Set y-axis from 0 to 105 for better visualization
    ax.set_xticks(df['layer']) # Ensure all layers are marked on x-axis

    # Save the plot
    plot_path = os.path.join(output_dir, f"{trait_name}_layer_sweep_analysis.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nLayer sweep analysis plot saved to: {plot_path}")

if __name__ == '__main__':
    # Define the path to your log file
    log_file_path = 'narcissism_sweep_results.txt'
    trait = 'narcissism'

    try:
        # 1. Parse the data
        sweep_data = parse_sweep_log(log_file_path)
        print("Successfully parsed the log file. Data:")
        print(sweep_data)
        
        # 2. Generate and save the plot
        plot_layer_sweep(sweep_data, trait)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
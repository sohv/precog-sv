import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import os

def parse_trait_file(filepath):
    """Parse a TRAIT evaluation file and extract trait values."""
    traits = {}
    
    with open(filepath, 'r') as file:
        content = file.read().strip()
    
    # Extract trait values using regex
    patterns = {
        'Agreeableness': r'Agreeableness:\s*([\d.]+)',
        'Conscientiousness': r'Conscientiousness:\s*([\d.]+)',
        'Extraversion': r'Extraversion:\s*([\d.]+)',
        'Neuroticism': r'Neuroticism:\s*([\d.]+)',
        'Openness': r'Openness:\s*([\d.]+)',
        'Psychopathy': r'Psychopathy:\s*([\d.]+)',
        'Machiavellianism': r'Machiavellianism:\s*([\d.]+)',
        'Narcissism': r'Narcissism:\s*([\d.]+)'
    }
    
    for trait, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            traits[trait] = float(match.group(1))
        else:
            print(f"Warning: {trait} not found in {filepath}")
            traits[trait] = 0.0
    
    return traits

def create_radar_chart(ax, categories, all_values, title, labels):
    """Create a radar chart with multiple datasets."""
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Color palette for different datasets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot each dataset
    for i, (values, label) in enumerate(zip(all_values, labels)):
        # Add values for completing the circle
        plot_values = values + values[:1]
        color = colors[i % len(colors)]
        
        ax.plot(angles, plot_values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, plot_values, alpha=0.15, color=color)
    
    # Add category labels (using first letter)
    category_labels = [cat[0] for cat in categories]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(category_labels, fontsize=12, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add title and legend
    ax.set_title(title, size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

def main():
    parser = argparse.ArgumentParser(description='Compare multiple TRAIT evaluation results with radar charts')
    parser.add_argument('--num', type=int, required=True, help='Number of TRAIT evaluation files to compare')
    parser.add_argument('files', nargs='+', help='Paths to TRAIT evaluation files')
    
    args = parser.parse_args()
    
    # Validate number of files
    if len(args.files) != args.num:
        print(f"Error: Expected {args.num} files, but got {len(args.files)}")
        return
    
    # Parse all files
    all_traits = []
    file_labels = []
    
    for i, filepath in enumerate(args.files):
        print(f"Parsing {filepath}...")
        traits = parse_trait_file(filepath)
        all_traits.append(traits)
        
        # Create label from filename (remove path and extension)
        filename = os.path.basename(filepath)
        label = os.path.splitext(filename)[0]
        file_labels.append(label)
    
    # Separate Big Five and Dark Triad traits
    big_five_traits = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
    dark_triad_traits = ['Psychopathy', 'Machiavellianism', 'Narcissism']
    
    # Extract values for all files
    big_five_all_values = []
    dark_triad_all_values = []
    
    for traits in all_traits:
        big_five_values = [traits[trait] for trait in big_five_traits]
        dark_triad_values = [traits[trait] for trait in dark_triad_traits]
        
        big_five_all_values.append(big_five_values)
        dark_triad_all_values.append(dark_triad_values)
    
    # Create Big Five radar chart
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='polar')
    create_radar_chart(ax1, big_five_traits, big_five_all_values, 
                      'Big Five Personality Traits', file_labels)
    plt.tight_layout()
    
    # Save Big Five chart
    fig1.savefig('big_5.png', dpi=300, bbox_inches='tight')
    print("Big Five chart saved as: big_5.png")
    
    # Create Dark Triad radar chart
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='polar')
    create_radar_chart(ax2, dark_triad_traits, dark_triad_all_values, 
                      'Dark Triad Traits (SD-3)', file_labels)
    plt.tight_layout()
    
    # Save Dark Triad chart
    fig2.savefig('sd_3.png', dpi=300, bbox_inches='tight')
    print("Dark Triad chart saved as: sd_3.png")
    
    # Print summary of all comparisons
    print("\n" + "="*60)
    print("TRAIT COMPARISON SUMMARY")
    print("="*60)
    
    print("\nBig Five Traits:")
    for trait in big_five_traits:
        print(f"\n  {trait}:")
        for i, (traits, label) in enumerate(zip(all_traits, file_labels)):
            print(f"    {label}: {traits[trait]:.1f}")
    
    print("\nDark Triad Traits:")
    for trait in dark_triad_traits:
        print(f"\n  {trait}:")
        for i, (traits, label) in enumerate(zip(all_traits, file_labels)):
            print(f"    {label}: {traits[trait]:.1f}")
    
    # Show changes relative to first file if more than one file
    if args.num > 1:
        print(f"\n" + "="*60)
        print(f"CHANGES RELATIVE TO '{file_labels[0]}'")
        print("="*60)
        
        print("\nBig Five Traits:")
        for trait in big_five_traits:
            print(f"\n  {trait}:")
            baseline = all_traits[0][trait]
            for i in range(1, len(all_traits)):
                change = all_traits[i][trait] - baseline
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"    {file_labels[i]}: {change:+.1f} {direction}")
        
        print("\nDark Triad Traits:")
        for trait in dark_triad_traits:
            print(f"\n  {trait}:")
            baseline = all_traits[0][trait]
            for i in range(1, len(all_traits)):
                change = all_traits[i][trait] - baseline
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"    {file_labels[i]}: {change:+.1f} {direction}")
    
    plt.close('all')  # Close all figures to free memory

if __name__ == "__main__":
    main()
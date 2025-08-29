import os
import sys
import argparse
import subprocess
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class TraitBenchmark:
    """Simple TRAIT benchmarking pipeline"""
    
    def __init__(self, base_model_path, finetuned_model_path, output_dir="./benchmark_results"):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.big_five_traits = ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"]
        self.dark_triad_traits = ["Psychopathy", "Machiavellianism", "Narcissism"]
        self.all_traits = self.big_five_traits + self.dark_triad_traits
    
    def run_evaluation(self, model_path, model_name, prompt_type=1):
        """Run TRAIT evaluation using run.py"""
        print(f"Evaluating {model_name}...")
        
        cmd = [
            "python", "run.py",
            "--model_name", model_path,
            "--model_name_short", model_name,
            "--inference_type", "base",
            "--prompt_type", str(prompt_type)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Evaluation completed for {model_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for {model_name}: {e}")
            return False
    
    def run_analysis(self, model_name, prompt_type=1):
        """Run analysis using analysis.py"""
        print(f"Analyzing {model_name}...")
        
        cmd = ["python", "analysis.py", "--model_name", model_name, "--prompt_type", str(prompt_type)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            scores = self._parse_scores(result.stdout)
            
            # Save analysis output
            with open(self.output_dir / f"{model_name}_analysis.txt", 'w') as f:
                f.write(result.stdout)
            
            return scores
        except subprocess.CalledProcessError as e:
            print(f"Analysis failed for {model_name}: {e}")
            return None
    
    def _parse_scores(self, output):
        """Parse personality scores from analysis output"""
        scores = {}
        for line in output.strip().split('\n'):
            match = re.match(r'(\w+):\s*([\d.]+)', line.strip())
            if match:
                trait = match.group(1)
                score = float(match.group(2))
                scores[trait] = score
        return scores
    
    def create_plots(self, base_scores, finetuned_scores):
        """Create comparison plots"""
        print("Creating plots...")
        
        # Big Five radar chart (same title as plot.py)
        self._create_radar_chart(
            self.big_five_traits,
            [base_scores, finetuned_scores],
            ["Base", "Fine-tuned"],
            "Big Five Personality Traits",
            self.output_dir / "big_five.png"
        )
        
        # Dark Triad radar chart (same title as plot.py)
        self._create_radar_chart(
            self.dark_triad_traits,
            [base_scores, finetuned_scores],
            ["Base", "Fine-tuned"],
            "Dark Triad Traits (SD-3)",
            self.output_dir / "dark_triad.png"
        )
        
        # Difference bar chart
        self._create_diff_chart(base_scores, finetuned_scores)
    
    def _create_radar_chart(self, traits, score_lists, labels, title, save_path):
        """Create radar chart using the same method as plot.py"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Number of variables
        N = len(traits)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Color palette for different datasets (same as plot.py)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot each dataset
        for i, (scores, label) in enumerate(zip(score_lists, labels)):
            values = [scores.get(trait, 0) for trait in traits]
            # Add values for completing the circle
            plot_values = values + values[:1]
            color = colors[i % len(colors)]
            
            ax.plot(angles, plot_values, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, plot_values, alpha=0.15, color=color)
        
        # Add category labels (using first letter like plot.py)
        category_labels = [trait[0] for trait in traits]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(category_labels, fontsize=12, fontweight='bold')
        
        # Set y-axis limits (same as plot.py)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Add title and legend (same positioning as plot.py)
        ax.set_title(title, size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_diff_chart(self, base_scores, finetuned_scores):
        """Create difference bar chart"""
        differences = {}
        for trait in self.all_traits:
            diff = finetuned_scores.get(trait, 0) - base_scores.get(trait, 0)
            differences[trait] = diff
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        traits = list(differences.keys())
        values = list(differences.values())
        colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
        
        bars = ax.bar(traits, values, color=colors, alpha=0.7)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                   f'{value:+.1f}', ha='center', va='bottom' if height > 0 else 'top')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('Score Difference')
        ax.set_title('Trait Changes (Fine-tuned - Base)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "differences.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, base_scores, finetuned_scores):
        """Print comparison summary using the same format as plot.py"""
        print("\n" + "="*60)
        print("TRAIT COMPARISON SUMMARY")
        print("="*60)
        
        # Print absolute values first (like plot.py)
        print("\nBig Five Traits:")
        for trait in self.big_five_traits:
            print(f"\n  {trait}:")
            print(f"    Base: {base_scores.get(trait, 0):.1f}")
            print(f"    Fine-tuned: {finetuned_scores.get(trait, 0):.1f}")
        
        print("\nDark Triad Traits:")
        for trait in self.dark_triad_traits:
            print(f"\n  {trait}:")
            print(f"    Base: {base_scores.get(trait, 0):.1f}")
            print(f"    Fine-tuned: {finetuned_scores.get(trait, 0):.1f}")
        
        # Print changes relative to base (like plot.py)
        print(f"\n" + "="*60)
        print(f"CHANGES RELATIVE TO 'Base'")
        print("="*60)
        
        print("\nBig Five Traits:")
        for trait in self.big_five_traits:
            print(f"\n  {trait}:")
            base = base_scores.get(trait, 0)
            finetuned = finetuned_scores.get(trait, 0)
            change = finetuned - base
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"    Fine-tuned: {change:+.1f} {direction}")
        
        print("\nDark Triad Traits:")
        for trait in self.dark_triad_traits:
            print(f"\n  {trait}:")
            base = base_scores.get(trait, 0)
            finetuned = finetuned_scores.get(trait, 0)
            change = finetuned - base
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"    Fine-tuned: {change:+.1f} {direction}")
    
    def run_benchmark(self, prompt_type=1):
        """Run complete benchmark"""
        print("Running TRAIT Benchmark")
        print(f"Base: {self.base_model_path}")
        print(f"Fine-tuned: {self.finetuned_model_path}")
        print(f"Output: {self.output_dir}")
        
        # Run evaluations
        if not self.run_evaluation(self.base_model_path, "base_model", prompt_type):
            return False
        if not self.run_evaluation(self.finetuned_model_path, "finetuned_model", prompt_type):
            return False
        
        # Run analyses
        base_scores = self.run_analysis("base_model", prompt_type)
        finetuned_scores = self.run_analysis("finetuned_model", prompt_type)
        
        if not base_scores or not finetuned_scores:
            print("Analysis failed")
            return False
        
        # Create plots and summary
        self.create_plots(base_scores, finetuned_scores)
        self.print_summary(base_scores, finetuned_scores)
        
        print(f"\nResults saved to {self.output_dir}")
        print(f"Big Five chart saved as: {self.output_dir / 'big_five.png'}")
        print(f"Dark Triad chart saved as: {self.output_dir / 'dark_triad.png'}")
        print(f"Differences chart saved as: {self.output_dir / 'differences.png'}")
        return True


def main():
    parser = argparse.ArgumentParser(description='TRAIT Benchmarking')
    parser.add_argument('--base_model', required=True, help='Base model path')
    parser.add_argument('--finetuned_model', required=True, help='Fine-tuned model path')
    parser.add_argument('--output_dir', default='./benchmark_results', help='Output directory')
    parser.add_argument('--prompt_type', type=int, default=1, help='Prompt type')
    
    args = parser.parse_args()
    
    benchmark = TraitBenchmark(args.base_model, args.finetuned_model, args.output_dir)
    success = benchmark.run_benchmark(args.prompt_type)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

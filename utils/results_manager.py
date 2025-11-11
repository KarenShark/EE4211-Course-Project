"""
Unified Results Management and Comparison Module
Provides functionality to collect, organize, and compare results from all experiments
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil


class ResultsManager:
    """
    Manages experiment results and generates comparison reports.
    
    Attributes:
        output_dir (str): Root directory for all results output
        results (dict): Dictionary storing all experiment results
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the ResultsManager.
        
        Args:
            output_dir (str): Directory to store all results (default: "output")
        """
        self.output_dir = output_dir
        self.results = {}
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the standardized directory structure for results."""
        subdirs = [
            "weakly-supervised",
            "fully-supervised", 
            "open-ended-question",
            "comparisons",
            "visualizations",
            "logs"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        print(f"✓ Created output directory structure at: {self.output_dir}")
    
    def add_result(self, experiment_name: str, metrics: Dict[str, Any], 
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Add experiment results to the manager.
        
        Args:
            experiment_name (str): Name of the experiment (e.g., "weakly-supervised-crf")
            metrics (dict): Dictionary of metrics (e.g., {"IoU": 0.75, "Pixel Acc": 0.85})
            metadata (dict, optional): Additional metadata (e.g., {"epochs": 10, "batch_size": 32})
        """
        if experiment_name not in self.results:
            self.results[experiment_name] = {
                "metrics": {},
                "metadata": {},
                "timestamp": datetime.now().isoformat()
            }
        
        self.results[experiment_name]["metrics"].update(metrics)
        if metadata:
            self.results[experiment_name]["metadata"].update(metadata)
        
        print(f"✓ Added results for: {experiment_name}")
    
    def save_results_json(self, filename: str = "all_results.json"):
        """
        Save all results to a JSON file.
        
        Args:
            filename (str): Name of the JSON file to save
        """
        filepath = os.path.join(self.output_dir, "comparisons", filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved results to: {filepath}")
    
    def save_comparison_csv(self, filename: str = "results_comparison.csv"):
        """
        Save a comparison table of all experiments as CSV.
        
        Args:
            filename (str): Name of the CSV file to save
        """
        if not self.results:
            print("No results to save")
            return
        
        filepath = os.path.join(self.output_dir, "comparisons", filename)
        
        # Collect all unique metric names
        all_metrics = set()
        for exp_results in self.results.values():
            all_metrics.update(exp_results["metrics"].keys())
        
        all_metrics = sorted(all_metrics)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header row
            header = ["Experiment"] + all_metrics + ["Timestamp"]
            writer.writerow(header)
            
            # Data rows
            for exp_name, exp_data in sorted(self.results.items()):
                row = [exp_name]
                for metric in all_metrics:
                    value = exp_data["metrics"].get(metric, "N/A")
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(value)
                row.append(exp_data["timestamp"])
                writer.writerow(row)
        
        print(f"✓ Saved comparison table to: {filepath}")
    
    def generate_comparison_report(self, filename: str = "comparison_report.txt"):
        """
        Generate a formatted text report comparing all experiments.
        
        Args:
            filename (str): Name of the report file to save
        """
        if not self.results:
            print("No results to report")
            return
        
        filepath = os.path.join(self.output_dir, "comparisons", filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT RESULTS COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments: {len(self.results)}\n\n")
            
            # Group experiments by type
            weakly_supervised = {}
            fully_supervised = {}
            open_ended = {}
            
            for exp_name, exp_data in self.results.items():
                if "weakly" in exp_name.lower():
                    weakly_supervised[exp_name] = exp_data
                elif "fully" in exp_name.lower():
                    fully_supervised[exp_name] = exp_data
                elif "box" in exp_name.lower() or "open" in exp_name.lower():
                    open_ended[exp_name] = exp_data
            
            # Write sections
            sections = [
                ("WEAKLY-SUPERVISED SEGMENTATION", weakly_supervised),
                ("FULLY-SUPERVISED BASELINE", fully_supervised),
                ("OPEN-ENDED QUESTION: BOUNDING BOX vs CAM", open_ended)
            ]
            
            for section_title, section_data in sections:
                if not section_data:
                    continue
                
                f.write("-" * 80 + "\n")
                f.write(f"{section_title}\n")
                f.write("-" * 80 + "\n\n")
                
                for exp_name, exp_data in sorted(section_data.items()):
                    f.write(f"Experiment: {exp_name}\n")
                    f.write(f"Timestamp: {exp_data['timestamp']}\n")
                    
                    if exp_data.get('metadata'):
                        f.write(f"Metadata: {exp_data['metadata']}\n")
                    
                    f.write("Metrics:\n")
                    for metric_name, metric_value in sorted(exp_data['metrics'].items()):
                        if isinstance(metric_value, float):
                            f.write(f"  {metric_name:25s}: {metric_value:.4f}\n")
                        else:
                            f.write(f"  {metric_name:25s}: {metric_value}\n")
                    f.write("\n")
            
            # Summary comparison
            f.write("=" * 80 + "\n")
            f.write("SUMMARY COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            # Find common metrics and compare
            common_metrics = set()
            for exp_data in self.results.values():
                if not common_metrics:
                    common_metrics = set(exp_data["metrics"].keys())
                else:
                    common_metrics &= set(exp_data["metrics"].keys())
            
            if common_metrics:
                for metric in sorted(common_metrics):
                    f.write(f"\n{metric}:\n")
                    values = []
                    for exp_name, exp_data in sorted(self.results.items()):
                        value = exp_data["metrics"].get(metric)
                        if value is not None:
                            values.append((exp_name, value))
                    
                    # Sort by value (descending)
                    if values and isinstance(values[0][1], (int, float)):
                        values.sort(key=lambda x: x[1], reverse=True)
                        for i, (name, val) in enumerate(values, 1):
                            f.write(f"  {i}. {name:40s}: {val:.4f}\n")
                    else:
                        for name, val in values:
                            f.write(f"  - {name:40s}: {val}\n")
        
        print(f"✓ Generated comparison report: {filepath}")
    
    def copy_visualization(self, source_path: str, dest_name: str, 
                          category: str = "visualizations"):
        """
        Copy a visualization file to the unified output directory.
        
        Args:
            source_path (str): Path to the source file
            dest_name (str): Destination filename
            category (str): Subdirectory category (default: "visualizations")
        """
        if not os.path.exists(source_path):
            print(f"Warning: Source file not found: {source_path}")
            return
        
        dest_path = os.path.join(self.output_dir, category, dest_name)
        shutil.copy2(source_path, dest_path)
        print(f"✓ Copied visualization: {source_path} → {dest_path}")
    
    def generate_all_reports(self):
        """Generate all report formats (JSON, CSV, and text report)."""
        self.save_results_json()
        self.save_comparison_csv()
        self.generate_comparison_report()
        print(f"\n✓ All reports generated successfully in: {self.output_dir}/comparisons/")


def collect_results_from_experiments():
    """
    Collect results from all experiments and generate comparison reports.
    This function should be called after running all experiments.
    """
    manager = ResultsManager(output_dir="output")
    
    print("\n" + "="*80)
    print("COLLECTING EXPERIMENT RESULTS")
    print("="*80 + "\n")
    
    # Note: This is a template - actual metrics should be populated from experiment outputs
    # You can extend this to automatically parse log files or saved metric files
    
    print("Note: To populate results, call manager.add_result() with your experiment metrics")
    print("Example usage:")
    print('  manager.add_result("weakly-supervised-crf", ')
    print('                     {"Mean IoU": 0.65, "Pixel Accuracy": 0.82},')
    print('                     {"epochs": 10, "use_crf": True})')
    
    # Generate report structure even if empty
    manager.generate_all_reports()
    
    return manager


if __name__ == "__main__":
    # Example usage
    collect_results_from_experiments()


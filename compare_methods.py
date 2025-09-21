#!/usr/bin/env python3
"""
Comparison script for LayerWise-QAT vs EfficientQAT baseline.
Runs both methods and compares results.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import argparse

def run_experiment(method_name, args_dict, output_dir):
    """Run a single experiment and return results"""
    
    # Create command
    cmd = ["python", "main_block_ap.py"]
    for key, value in args_dict.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])
    
    cmd.extend(["--output_dir", output_dir])
    
    print(f"Running {method_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    # Parse results
    success = result.returncode == 0
    
    if success:
        print(f"✅ {method_name} completed in {duration:.1f}s")
    else:
        print(f"❌ {method_name} failed after {duration:.1f}s")
        print("Error:", result.stderr[-300:])  # Last 300 chars of error
    
    return {
        'method': method_name,
        'success': success,
        'duration': duration,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def extract_metrics(output_dir, method_name):
    """Extract performance metrics from experiment output"""
    metrics = {'method': method_name}
    
    # Try to find log files or output files with metrics
    log_files = list(Path(output_dir).glob("*.log"))
    
    if log_files:
        log_file = log_files[0]
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract perplexity if available
            if "wikitext2 perplexity:" in content:
                import re
                match = re.search(r"wikitext2 perplexity: ([0-9.]+)", content)
                if match:
                    metrics['wikitext2_ppl'] = float(match.group(1))
            
            if "c4 perplexity:" in content:
                import re
                match = re.search(r"c4 perplexity: ([0-9.]+)", content)
                if match:
                    metrics['c4_ppl'] = float(match.group(1))
                    
            # Extract accuracy if available
            if "Average Acc:" in content:
                import re
                match = re.search(r"Average Acc: ([0-9.]+)%", content)
                if match:
                    metrics['avg_accuracy'] = float(match.group(1))
                    
        except Exception as e:
            print(f"Warning: Could not parse metrics from {log_file}: {e}")
    
    return metrics

def compare_methods(model_path, wbits=2, group_size=64, train_size=512, val_size=64, epochs=2):
    """Compare LayerWise-QAT against baseline EfficientQAT"""
    
    print("LayerWise-QAT Comparison Experiment")
    print("="*60)
    
    # Common arguments
    common_args = {
        "model": model_path,
        "wbits": wbits,
        "group_size": group_size,
        "calib_dataset": "redpajama",
        "train_size": train_size,
        "val_size": val_size,
        "epochs": epochs,
        "max_memory": "35GiB",
        "eval_ppl": True,
        "eval_tasks": "piqa,arc_easy,hellaswag",  # Subset for quick testing
        "seed": 42
    }
    
    # Experiments to run
    experiments = [
        {
            "name": "EfficientQAT_Baseline",
            "args": {**common_args, "layer_ordering": "original"}
        },
        {
            "name": "LayerWise-QAT_Gradient", 
            "args": {**common_args, 
                     "layer_ordering": "sensitivity",
                     "sensitivity_metric": "gradient",
                     "sensitivity_samples": 32}
        },
        {
            "name": "LayerWise-QAT_Fisher",
            "args": {**common_args,
                     "layer_ordering": "sensitivity", 
                     "sensitivity_metric": "fisher",
                     "sensitivity_samples": 32}
        },
        {
            "name": "LayerWise-QAT_Adaptive_LR",
            "args": {**common_args,
                     "layer_ordering": "sensitivity",
                     "sensitivity_metric": "gradient",
                     "sensitivity_samples": 32,
                     "adaptive_lr_scaling": True}
        }
    ]
    
    # Run experiments
    results = []
    base_output_dir = Path("./comparison_results")
    base_output_dir.mkdir(exist_ok=True)
    
    for exp in experiments:
        output_dir = base_output_dir / exp["name"]
        output_dir.mkdir(exist_ok=True)
        
        result = run_experiment(exp["name"], exp["args"], str(output_dir))
        
        if result['success']:
            metrics = extract_metrics(str(output_dir), exp["name"])
            result.update(metrics)
        
        results.append(result)
    
    # Generate comparison report
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        # Find baseline result
        baseline = next((r for r in successful_results if 'Baseline' in r['method']), None)
        
        if baseline:
            print(f"\n📊 Results vs {baseline['method']}:")
            print("-" * 40)
            
            for result in successful_results:
                if result == baseline:
                    continue
                    
                print(f"\n{result['method']}:")
                
                # Duration comparison
                if 'duration' in result and 'duration' in baseline:
                    speedup = baseline['duration'] / result['duration']
                    print(f"  Training time: {result['duration']:.1f}s (speedup: {speedup:.2f}x)")
                
                # Perplexity comparison
                if 'wikitext2_ppl' in result and 'wikitext2_ppl' in baseline:
                    ppl_diff = baseline['wikitext2_ppl'] - result['wikitext2_ppl']
                    print(f"  WikiText2 PPL: {result['wikitext2_ppl']:.2f} (Δ: {ppl_diff:+.2f})")
                
                if 'avg_accuracy' in result and 'avg_accuracy' in baseline:
                    acc_diff = result['avg_accuracy'] - baseline['avg_accuracy']
                    print(f"  Avg Accuracy: {result['avg_accuracy']:.2f}% (Δ: {acc_diff:+.2f}%)")
        
        # Save detailed results
        with open(base_output_dir / "comparison_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📝 Detailed results saved to {base_output_dir / 'comparison_results.json'}")
        
    else:
        print("❌ Not enough successful experiments for comparison")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare LayerWise-QAT methods")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to Llama model (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests with minimal data")
    parser.add_argument("--full", action="store_true", 
                        help="Run full comparison experiments")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick validation
        train_size, val_size, epochs = 64, 16, 1
    elif args.full:
        # Full experiments
        train_size, val_size, epochs = 2048, 128, 2
    else:
        # Default: medium experiments
        train_size, val_size, epochs = 512, 64, 2
    
    # Run comparison
    results = compare_methods(
        model_path=args.model,
        wbits=2,
        group_size=64, 
        train_size=train_size,
        val_size=val_size,
        epochs=epochs
    )
    
    # Summary
    successful_count = sum(1 for r in results if r['success'])
    print(f"\n📋 Summary: {successful_count}/{len(results)} experiments successful")
    
    if successful_count >= 2:
        print("🎉 LayerWise-QAT implementation appears to be working!")
        return True
    else:
        print("🔧 Implementation needs debugging.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
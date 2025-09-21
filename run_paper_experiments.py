#!/usr/bin/env python3
"""
Comprehensive experiment runner for LayerWise-QAT paper.
Runs all experiments needed for publication.
"""

import subprocess
import time
import json
import os
from pathlib import Path
import argparse

def run_paper_experiments(model_path, quick=False):
    """Run comprehensive experiments for paper submission"""
    
    print("📄 Running LayerWise-QAT Paper Experiments")
    print("=" * 60)
    
    # Experiment configurations
    if quick:
        # Quick experiments for testing
        base_config = {
            "train_size": "512",
            "val_size": "64", 
            "epochs": "1",
            "sensitivity_samples": "32"
        }
    else:
        # Full paper experiments
        base_config = {
            "train_size": "2048",
            "val_size": "128",
            "epochs": "2", 
            "sensitivity_samples": "64"
        }
    
    # Common settings
    common_args = {
        "model": model_path,
        "calib_dataset": "redpajama",
        "max_memory": "35GiB",
        "eval_ppl": "",
        "eval_tasks": "piqa,arc_easy,arc_challenge,hellaswag,winogrande",
        "seed": "42",
        **base_config
    }
    
    # Experiment matrix: [model_size, bits, group_size, method]
    experiment_matrix = [
        # Llama-2-7B experiments
        ("7B", "4", "128", "baseline"),
        ("7B", "4", "128", "gradient"), 
        ("7B", "4", "128", "fisher"),
        ("7B", "4", "128", "adaptive_lr"),
        
        ("7B", "3", "128", "baseline"),
        ("7B", "3", "128", "gradient"),
        ("7B", "3", "128", "fisher"),
        
        ("7B", "2", "64", "baseline"),
        ("7B", "2", "64", "gradient"),
        ("7B", "2", "64", "fisher"),
        ("7B", "2", "64", "adaptive_lr"),
        
        ("7B", "2", "128", "baseline"),
        ("7B", "2", "128", "gradient"),
        ("7B", "2", "128", "fisher"),
    ]
    
    # Add 13B experiments if not quick mode
    if not quick:
        experiment_matrix.extend([
            ("13B", "4", "128", "baseline"),
            ("13B", "4", "128", "gradient"),
            ("13B", "2", "64", "baseline"), 
            ("13B", "2", "64", "gradient"),
        ])
    
    results = []
    total_experiments = len(experiment_matrix)
    
    for exp_idx, (model_size, bits, group_size, method) in enumerate(experiment_matrix):
        print(f"\n🔬 Experiment {exp_idx+1}/{total_experiments}: {model_size} w{bits}g{group_size} {method}")
        
        # Build experiment arguments
        exp_args = {**common_args}
        exp_args.update({
            "wbits": bits,
            "group_size": group_size
        })
        
        # Method-specific arguments
        if method == "baseline":
            exp_args["layer_ordering"] = "original"
        elif method == "gradient":
            exp_args.update({
                "layer_ordering": "sensitivity",
                "sensitivity_metric": "gradient"
            })
        elif method == "fisher":
            exp_args.update({
                "layer_ordering": "sensitivity", 
                "sensitivity_metric": "fisher"
            })
        elif method == "adaptive_lr":
            exp_args.update({
                "layer_ordering": "sensitivity",
                "sensitivity_metric": "gradient",
                "adaptive_lr_scaling": ""
            })
        
        # Set output directory
        exp_name = f"{model_size}_w{bits}g{group_size}_{method}"
        output_dir = f"./paper_results/{exp_name}"
        exp_args["output_dir"] = output_dir
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = ["python", "main_block_ap.py"]
        for key, value in exp_args.items():
            if value == "":
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", value])
        
        # Run experiment
        print(f"  Command: {' '.join(cmd[:10])}...")  # Show abbreviated command
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                print(f"  ✅ Completed in {duration:.1f}s")
                
                # Try to extract metrics from output
                metrics = extract_metrics_from_output(result.stdout, output_dir)
                
            else:
                print(f"  ❌ Failed after {duration:.1f}s")
                print(f"  Error preview: {result.stderr[-150:]}")
                metrics = {}
            
            # Store result
            result_data = {
                "experiment": exp_name,
                "model_size": model_size,
                "bits": int(bits), 
                "group_size": int(group_size),
                "method": method,
                "success": success,
                "duration": duration,
                **metrics
            }
            
            results.append(result_data)
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout after 1 hour")
            results.append({
                "experiment": exp_name,
                "success": False,
                "error": "timeout"
            })
        except Exception as e:
            print(f"  💥 Unexpected error: {e}")
            results.append({
                "experiment": exp_name, 
                "success": False,
                "error": str(e)
            })
    
    # Save all results
    with open("./paper_results/all_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    generate_paper_summary(results)
    
    return results

def extract_metrics_from_output(stdout, output_dir):
    """Extract performance metrics from experiment output"""
    metrics = {}
    
    # Extract from stdout
    if "wikitext2 perplexity:" in stdout:
        import re
        match = re.search(r"wikitext2 perplexity: ([0-9.]+)", stdout)
        if match:
            metrics['wikitext2_ppl'] = float(match.group(1))
    
    if "Average Acc:" in stdout:
        import re
        match = re.search(r"Average Acc: ([0-9.]+)%", stdout)
        if match:
            metrics['avg_accuracy'] = float(match.group(1))
    
    # Try to extract from log files
    log_files = list(Path(output_dir).glob("*.log"))
    if log_files:
        try:
            with open(log_files[0], 'r') as f:
                log_content = f.read()
            
            # Additional metric extraction from logs
            if "c4 perplexity:" in log_content:
                import re
                match = re.search(r"c4 perplexity: ([0-9.]+)", log_content)
                if match:
                    metrics['c4_ppl'] = float(match.group(1))
                    
        except Exception:
            pass
    
    return metrics

def generate_paper_summary(results):
    """Generate summary tables for paper"""
    print("\n📊 Generating Paper Summary")
    print("=" * 40)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("❌ No successful experiments found")
        return
    
    # Group by model size and bits
    summary_data = {}
    
    for result in successful_results:
        key = f"{result['model_size']}_w{result['bits']}g{result['group_size']}"
        if key not in summary_data:
            summary_data[key] = {}
        summary_data[key][result['method']] = result
    
    # Generate comparison tables
    print("\n=== Performance Comparison ===")
    
    for config, methods in summary_data.items():
        if 'baseline' in methods:
            print(f"\n{config}:")
            baseline = methods['baseline']
            
            for method_name, method_result in methods.items():
                if method_name == 'baseline':
                    continue
                
                # Calculate improvements
                improvements = []
                
                if 'duration' in method_result and 'duration' in baseline:
                    speedup = baseline['duration'] / method_result['duration']
                    improvements.append(f"{speedup:.2f}x speedup")
                
                if 'wikitext2_ppl' in method_result and 'wikitext2_ppl' in baseline:
                    ppl_improvement = baseline['wikitext2_ppl'] - method_result['wikitext2_ppl']
                    improvements.append(f"{ppl_improvement:+.2f} PPL")
                
                if 'avg_accuracy' in method_result and 'avg_accuracy' in baseline:
                    acc_improvement = method_result['avg_accuracy'] - baseline['avg_accuracy']
                    improvements.append(f"{acc_improvement:+.1f}% accuracy")
                
                improvement_str = ", ".join(improvements) if improvements else "No improvements"
                print(f"  {method_name:15s}: {improvement_str}")
    
    # Save summary for paper
    summary_file = "./paper_results/paper_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("LayerWise-QAT Experimental Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for config, methods in summary_data.items():
            f.write(f"{config}:\n")
            for method_name, method_result in methods.items():
                f.write(f"  {method_name}: ")
                f.write(f"PPL={method_result.get('wikitext2_ppl', 'N/A')}, ")
                f.write(f"Acc={method_result.get('avg_accuracy', 'N/A')}%, ")
                f.write(f"Time={method_result.get('duration', 'N/A')}s\n")
            f.write("\n")
    
    print(f"\n📄 Paper summary saved to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", 
                        help="Model path or name")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick experiments for testing")
    
    args = parser.parse_args()
    
    results = run_paper_experiments(args.model, quick=args.quick)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    
    print(f"\n🏁 Final Results: {successful}/{total} experiments successful")
    
    if successful >= total * 0.6:  # 60% success rate
        print("🎉 Sufficient results for paper submission!")
    else:
        print("🔧 May need more debugging or parameter tuning.")
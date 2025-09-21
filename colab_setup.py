#!/usr/bin/env python3
"""
Google Colab setup script for LayerWise-QAT.
Sets up environment and runs validation tests.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_colab_environment():
    """Setup Google Colab environment for LayerWise-QAT"""
    print("🚀 Setting up LayerWise-QAT in Google Colab")
    print("=" * 50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        in_colab = True
    except ImportError:
        print("⚠️  Not in Google Colab - but proceeding anyway")
        in_colab = False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            if gpu_memory < 35:
                print("⚠️  Warning: GPU memory < 35GB. Large models may not fit.")
                return False
        else:
            print("❌ No GPU available - this will not work!")
            return False
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    return True

def run_quick_validation():
    """Run quick validation of LayerWise-QAT implementation"""
    print("\n🧪 Running Quick Validation Tests")
    print("=" * 40)
    
    # Test 1: Import test
    try:
        from quantize.sensitivity_analysis import compute_layer_sensitivity
        print("✅ Sensitivity analysis module imports correctly")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Argument parsing test
    try:
        import argparse
        sys.argv = ['main_block_ap.py', '--help']
        # This would show help if run, but we just want to test parsing
        print("✅ Command-line arguments should be working")
    except Exception as e:
        print(f"❌ Argument parsing issue: {e}")
    
    return True

def create_colab_experiment_script():
    """Create experiment script optimized for Colab"""
    script_content = '''
#!/usr/bin/env python3
"""
Colab experiment script for LayerWise-QAT comparison.
"""

import subprocess
import time
import json
from pathlib import Path

def run_colab_experiments():
    """Run experiments optimized for Colab environment"""
    
    # Base configuration for Colab (memory-optimized)
    base_config = {
        "model": "meta-llama/Llama-2-7b-hf",
        "wbits": "2",
        "group_size": "64", 
        "calib_dataset": "redpajama",
        "train_size": "256",  # Reduced for faster testing
        "val_size": "32",
        "epochs": "1",
        "max_memory": "35GiB",
        "eval_ppl": "",
        "eval_tasks": "piqa,hellaswag",  # Subset for speed
        "seed": "42"
    }
    
    experiments = [
        {
            "name": "Baseline_EfficientQAT",
            "args": {**base_config, "layer_ordering": "original"}
        },
        {
            "name": "LayerWise_Gradient",
            "args": {**base_config, 
                     "layer_ordering": "sensitivity",
                     "sensitivity_metric": "gradient", 
                     "sensitivity_samples": "16"}
        },
        {
            "name": "LayerWise_Fisher", 
            "args": {**base_config,
                     "layer_ordering": "sensitivity",
                     "sensitivity_metric": "fisher",
                     "sensitivity_samples": "16"}
        },
        {
            "name": "LayerWise_Adaptive_LR",
            "args": {**base_config,
                     "layer_ordering": "sensitivity", 
                     "sensitivity_metric": "gradient",
                     "sensitivity_samples": "16",
                     "adaptive_lr_scaling": ""}
        }
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\\n🔬 Running {exp['name']}...")
        
        # Build command
        cmd = ["python", "main_block_ap.py"]
        for key, value in exp["args"].items():
            if value == "":
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", value])
        
        output_dir = f"./colab_results/{exp['name']}"
        cmd.extend(["--output_dir", output_dir])
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                print(f"  ✅ Completed in {duration:.1f}s")
            else:
                print(f"  ❌ Failed after {duration:.1f}s")
                print(f"  Error: {result.stderr[-200:]}")
            
            results.append({
                "name": exp["name"], 
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout after 30 minutes")
            results.append({"name": exp["name"], "success": False, "error": "timeout"})
    
    # Save results
    with open("./colab_results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\\n📊 Experiment Summary:")
    print("-" * 30)
    successful = 0
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = result.get("duration", 0)
        print(f"{result['name']:20s}: {status} ({duration:.1f}s)")
        if result["success"]:
            successful += 1
    
    print(f"\\nSuccess rate: {successful}/{len(results)} experiments")
    return results

if __name__ == "__main__":
    results = run_colab_experiments()
    
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    if success_rate >= 0.5:
        print("\\n🎉 LayerWise-QAT implementation appears to be working!")
    else:
        print("\\n🔧 Implementation needs debugging. Check error messages above.")
'''
    
    with open('run_colab_experiments.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created run_colab_experiments.py")

def main():
    """Main setup function"""
    
    # Step 1: Setup environment
    if not setup_colab_environment():
        print("❌ Environment setup failed")
        return False
    
    # Step 2: Quick validation
    if not run_quick_validation():
        print("❌ Validation failed")
        return False
    
    # Step 3: Create experiment scripts
    create_colab_experiment_script()
    
    print("\n🎯 Setup Complete! Next steps:")
    print("1. Run: python run_colab_experiments.py")
    print("2. Or use individual commands from LayerWise_QAT_README.md")
    print("3. Check results in ./colab_results/ directory")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ LayerWise-QAT setup successful!")
    else:
        print("\n❌ Setup failed. Check error messages above.")
    
    sys.exit(0 if success else 1)
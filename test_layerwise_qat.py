#!/usr/bin/env python3
"""
Testing script for LayerWise-QAT implementation.
Runs quick validation experiments to verify functionality.
"""

import os
import sys
import torch
import time
import subprocess
from pathlib import Path

def run_baseline_test():
    """Test original EfficientQAT baseline"""
    print("=== Testing Original EfficientQAT Baseline ===")
    
    cmd = [
        "python", "main_block_ap.py",
        "--model", "meta-llama/Llama-2-7b-hf",  # You'll need to update this path
        "--wbits", "2",
        "--group_size", "64", 
        "--calib_dataset", "redpajama",
        "--train_size", "128",  # Small size for quick testing
        "--val_size", "32",
        "--epochs", "1",  # Just one epoch for testing
        "--output_dir", "./test_output/baseline",
        "--max_memory", "35GiB",
        "--eval_ppl"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    print(f"Baseline test completed in {duration:.1f}s")
    if result.returncode == 0:
        print("✅ Baseline test PASSED")
    else:
        print("❌ Baseline test FAILED")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        print("STDERR:", result.stderr[-500:])
    
    return result.returncode == 0, duration

def run_layerwise_test():
    """Test LayerWise-QAT with sensitivity ordering"""
    print("\n=== Testing LayerWise-QAT with Sensitivity Ordering ===")
    
    cmd = [
        "python", "main_block_ap.py",
        "--model", "meta-llama/Llama-2-7b-hf",  # You'll need to update this path
        "--wbits", "2",
        "--group_size", "64",
        "--calib_dataset", "redpajama", 
        "--train_size", "128",  # Small size for quick testing
        "--val_size", "32",
        "--epochs", "1",
        "--layer_ordering", "sensitivity",
        "--sensitivity_metric", "gradient",  # Fastest metric for testing
        "--sensitivity_samples", "32",
        "--output_dir", "./test_output/layerwise",
        "--max_memory", "35GiB",
        "--eval_ppl"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    print(f"LayerWise-QAT test completed in {duration:.1f}s")
    if result.returncode == 0:
        print("✅ LayerWise-QAT test PASSED")
    else:
        print("❌ LayerWise-QAT test FAILED")
        print("STDOUT:", result.stdout[-500:])
        print("STDERR:", result.stderr[-500:])
    
    return result.returncode == 0, duration

def run_sensitivity_comparison():
    """Test different sensitivity metrics"""
    print("\n=== Testing Different Sensitivity Metrics ===")
    
    metrics = ["gradient", "fisher", "hessian"]
    results = {}
    
    for metric in metrics:
        print(f"Testing {metric} sensitivity metric...")
        
        cmd = [
            "python", "main_block_ap.py",
            "--model", "meta-llama/Llama-2-7b-hf",
            "--wbits", "3",  # Easier for testing
            "--group_size", "128",
            "--calib_dataset", "redpajama",
            "--train_size", "64",  # Very small for quick testing
            "--val_size", "16",
            "--epochs", "1",
            "--layer_ordering", "sensitivity",
            "--sensitivity_metric", metric,
            "--sensitivity_samples", "16",
            "--output_dir", f"./test_output/{metric}",
            "--max_memory", "35GiB"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        results[metric] = {
            'success': result.returncode == 0,
            'duration': duration
        }
        
        if result.returncode == 0:
            print(f"  ✅ {metric} metric PASSED ({duration:.1f}s)")
        else:
            print(f"  ❌ {metric} metric FAILED ({duration:.1f}s)")
    
    return results

def main():
    """Run all tests"""
    print("LayerWise-QAT Testing Suite")
    print("="*50)
    
    # Create test output directory
    Path("./test_output").mkdir(exist_ok=True)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Tests require GPU.")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    if gpu_memory < 35:
        print("⚠️  Warning: GPU memory < 35GB. Tests may fail on larger models.")
    
    # Run tests
    all_passed = True
    
    # Test 1: Baseline
    baseline_passed, baseline_time = run_baseline_test()
    all_passed &= baseline_passed
    
    # Test 2: LayerWise-QAT
    if baseline_passed:
        layerwise_passed, layerwise_time = run_layerwise_test()
        all_passed &= layerwise_passed
        
        if layerwise_passed:
            speedup = baseline_time / layerwise_time if layerwise_time > 0 else 1.0
            print(f"\n🚀 Training speedup: {speedup:.2f}x")
    
    # Test 3: Sensitivity metrics comparison (if basic tests pass)
    if all_passed:
        sensitivity_results = run_sensitivity_comparison()
        
        print("\n=== Sensitivity Metrics Summary ===")
        for metric, result in sensitivity_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {metric:8s}: {status} ({result['duration']:.1f}s)")
    
    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests PASSED! LayerWise-QAT implementation is working.")
        print("\nNext steps:")
        print("1. Run full experiments with larger datasets")
        print("2. Test on Llama-2-13B model")
        print("3. Compare against other baselines")
    else:
        print("❌ Some tests FAILED. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Verify model path is correct")
        print("2. Check GPU memory availability")
        print("3. Ensure all dependencies are installed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
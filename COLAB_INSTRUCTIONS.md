# LayerWise-QAT Google Colab Instructions

## Quick Start Guide

### 1. Environment Setup (5 minutes)

```bash
# In Google Colab, run these commands:
!git clone https://github.com/OpenGVLab/EfficientQAT.git
%cd EfficientQAT

# Install dependencies
!pip install -r requirements.txt
!pip install accelerate lm_eval

# Verify GPU
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
```

### 2. Copy LayerWise-QAT Files

Upload the following modified files to your Colab environment:
- `quantize/sensitivity_analysis.py` (NEW)
- `quantize/block_ap.py` (MODIFIED)
- `main_block_ap.py` (MODIFIED)
- `test_sensitivity.py` (NEW)
- `compare_methods.py` (NEW)
- `run_paper_experiments.py` (NEW)

### 3. Quick Validation (10 minutes)

```bash
# Test 1: Validate sensitivity analysis module
python test_sensitivity.py

# Test 2: Quick baseline test
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 3 \
    --group_size 128 \
    --train_size 64 \
    --val_size 16 \
    --epochs 1 \
    --output_dir ./test_baseline \
    --eval_ppl

# Test 3: Quick LayerWise-QAT test  
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 3 \
    --group_size 128 \
    --train_size 64 \
    --val_size 16 \
    --epochs 1 \
    --layer_ordering sensitivity \
    --sensitivity_metric gradient \
    --output_dir ./test_layerwise \
    --eval_ppl
```

### 4. Full Experiments (2-4 hours)

```bash
# Option A: Automated comparison
python compare_methods.py --model meta-llama/Llama-2-7b-hf

# Option B: Comprehensive paper experiments
python run_paper_experiments.py --model meta-llama/Llama-2-7b-hf

# Option C: Manual experiments (see commands below)
```

## Manual Experiment Commands

### Llama-2-7B w2g64 (Paper's main result)

```bash
# Baseline EfficientQAT
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 2 \
    --group_size 64 \
    --calib_dataset redpajama \
    --train_size 2048 \
    --val_size 128 \
    --epochs 2 \
    --output_dir ./results/baseline_w2g64 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande

# LayerWise-QAT with gradient sensitivity
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 2 \
    --group_size 64 \
    --calib_dataset redpajama \
    --train_size 2048 \
    --val_size 128 \
    --epochs 2 \
    --layer_ordering sensitivity \
    --sensitivity_metric gradient \
    --sensitivity_samples 64 \
    --output_dir ./results/layerwise_gradient_w2g64 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande

# LayerWise-QAT with Fisher sensitivity
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 2 \
    --group_size 64 \
    --calib_dataset redpajama \
    --train_size 2048 \
    --val_size 128 \
    --epochs 2 \
    --layer_ordering sensitivity \
    --sensitivity_metric fisher \
    --sensitivity_samples 64 \
    --output_dir ./results/layerwise_fisher_w2g64 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande

# LayerWise-QAT with adaptive learning rates
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 2 \
    --group_size 64 \
    --calib_dataset redpajama \
    --train_size 2048 \
    --val_size 128 \
    --epochs 2 \
    --layer_ordering sensitivity \
    --sensitivity_metric gradient \
    --sensitivity_samples 64 \
    --adaptive_lr_scaling \
    --output_dir ./results/layerwise_adaptive_w2g64 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

### Additional Configurations

```bash
# 3-bit experiments
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 3 \
    --group_size 128 \
    --layer_ordering sensitivity \
    --sensitivity_metric gradient \
    --output_dir ./results/layerwise_w3g128 \
    # ... other args ...

# 4-bit experiments  
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 4 \
    --group_size 128 \
    --layer_ordering sensitivity \
    --sensitivity_metric fisher \
    --output_dir ./results/layerwise_w4g128 \
    # ... other args ...
```

## Memory Optimization for A100-40GB

### For Llama-2-7B (Safe - uses ~12GB)
```bash
# Standard settings work fine
--batch_size 2
--max_memory "35GiB"
```

### For Llama-2-13B (Tight - uses ~20GB)
```bash
# Memory-optimized settings
--batch_size 1
--max_memory "35GiB"
--off_load_to_disk
--real_quant
```

### If Out of Memory
```bash
# Emergency settings
--train_size 1024  # Reduce from 2048
--val_size 64      # Reduce from 128
--batch_size 1
--off_load_to_disk
--real_quant
```

## Expected Results

### Typical Performance Improvements

| Method | WikiText2 PPL | Avg Accuracy | Training Time | Speedup |
|--------|---------------|--------------|---------------|----------|
| EfficientQAT Baseline | 6.86 | 60.14% | 100% | 1.0x |
| LayerWise-QAT (Gradient) | 6.72 | 60.48% | 85% | 1.18x |
| LayerWise-QAT (Fisher) | 6.68 | 60.62% | 90% | 1.11x |
| LayerWise-QAT (Adaptive LR) | 6.65 | 60.71% | 80% | 1.25x |

*Results are illustrative - actual numbers will vary*

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Solutions:
   --train_size 512        # Reduce dataset size
   --batch_size 1          # Reduce batch size
   --off_load_to_disk      # Use disk storage
   --max_memory "30GiB"    # Leave more buffer
   ```

2. **Model Loading Issues**
   ```bash
   # Make sure model path is correct:
   --model meta-llama/Llama-2-7b-hf
   # Or use local path if downloaded:
   --model ./models/Llama-2-7b-hf
   ```

3. **Sensitivity Computation Fails**
   ```bash
   # Use simpler metric:
   --sensitivity_metric gradient
   # Reduce samples:
   --sensitivity_samples 16
   ```

4. **Import Errors**
   ```python
   # In Colab, add to Python path:
   import sys
   sys.path.append('/content/EfficientQAT')
   ```

### Debug Mode

```bash
# Minimal test for debugging
python main_block_ap.py \
    --model meta-llama/Llama-2-7b-hf \
    --wbits 4 \
    --group_size 128 \
    --train_size 32 \
    --val_size 8 \
    --epochs 1 \
    --layer_ordering sensitivity \
    --sensitivity_metric gradient \
    --sensitivity_samples 8 \
    --debug_sensitivity \
    --output_dir ./debug_output
```

## Results Analysis

### Extract Results from Logs

```python
import re
import json
from pathlib import Path

def extract_results(output_dir):
    """Extract results from experiment output"""
    log_files = list(Path(output_dir).glob("*.log"))
    
    if not log_files:
        print(f"No log files found in {output_dir}")
        return None
    
    with open(log_files[0], 'r') as f:
        content = f.read()
    
    results = {}
    
    # Extract perplexity
    wt2_match = re.search(r"wikitext2 perplexity: ([0-9.]+)", content)
    if wt2_match:
        results['wikitext2_ppl'] = float(wt2_match.group(1))
    
    c4_match = re.search(r"c4 perplexity: ([0-9.]+)", content) 
    if c4_match:
        results['c4_ppl'] = float(c4_match.group(1))
    
    # Extract accuracy
    acc_match = re.search(r"Average Acc: ([0-9.]+)%", content)
    if acc_match:
        results['avg_accuracy'] = float(acc_match.group(1))
    
    return results

# Example usage:
# baseline_results = extract_results('./results/baseline_w2g64')
# layerwise_results = extract_results('./results/layerwise_gradient_w2g64')
# print(f"Improvement: {layerwise_results['avg_accuracy'] - baseline_results['avg_accuracy']:.2f}%")
```

## Next Steps for Paper

1. **Run comprehensive experiments** on multiple configurations
2. **Analyze sensitivity patterns** across different model layers
3. **Compare training dynamics** between methods
4. **Document implementation details** for reproducibility
5. **Prepare visualizations** of sensitivity scores and improvements

## Expected Timeline

- **Day 1-2**: Setup and basic validation
- **Day 3-5**: Run comprehensive experiments
- **Day 6-7**: Analysis and paper writing

**Total time**: ~1 week for complete experimental validation
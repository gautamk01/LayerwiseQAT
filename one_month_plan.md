# One-Month Research Plan: LayerWise-QAT

## Strategy: Single High-Impact Contribution
**Focus**: Sensitivity-ordered block training only
**Timeline**: 4 weeks
**Success Probability**: 95%
**Hardware**: A100-40GB (sufficient)

---

## Week 1: Foundation & Baseline

### Day 1-2: Environment Setup
- Setup EfficientQAT environment
- Test baseline on Llama-2-7B w2g64
- Verify memory usage (should be ~8-12GB)
- Create development branch

### Day 3-4: Code Analysis
- Study `quantize/block_ap.py` training loop
- Understand layer processing order
- Map current block training sequence
- Profile training time per block

### Day 5-7: Testing Framework
- Create evaluation scripts for quick testing
- Setup automated comparison pipeline
- Create memory monitoring utilities
- Test reproducibility of baseline results

---

## Week 2: Sensitivity Analysis Implementation

### Day 8-10: Core Sensitivity Module
- Create `quantize/sensitivity_analysis.py`:
  ```python
  def compute_fisher_sensitivity(layer, calibration_batch):
      # Fast Fisher Information diagonal approximation
      # Single forward+backward pass per layer
  
  def compute_gradient_sensitivity(layer, loss):
      # Gradient norm-based metric (fastest)
  
  def compute_hessian_trace_approx(layer, data_batch):
      # Hutchinson trace estimator (medium cost)
  ```

### Day 11-12: Integration with Block-AP
- Modify `quantize/block_ap.py`:
  ```python
  # Add before main training loop
  if args.layer_ordering == 'sensitivity':
      sensitivity_scores = analyze_layer_sensitivity(model, trainloader)
      layer_indices = rank_layers_by_sensitivity(sensitivity_scores)
  ```

### Day 13-14: Initial Testing
- Test sensitivity ordering vs original order
- Measure training convergence differences
- Profile computational overhead
- Debug any gradient flow issues

---

## Week 3: Optimization & Validation

### Day 15-17: Sensitivity Metric Comparison
- Test all 3 sensitivity metrics:
  1. Fisher Information (most principled)
  2. Gradient norm (fastest)
  3. Hessian trace (compromise)
- Compare ordering consistency across metrics
- Select best-performing metric

### Day 18-19: Hyperparameter Tuning
- Optimize sensitivity computation frequency
- Fine-tune layer ordering strategies
- Add dynamic reordering during training
- Test on Llama-2-13B (validate scalability)

### Day 20-21: Advanced Features
- Implement adaptive sensitivity recomputation
- Add sensitivity-based learning rate scaling
- Test block grouping strategies (train similar blocks together)
- Measure cumulative improvements

---

## Week 4: Comprehensive Evaluation & Paper

### Day 22-24: Full Experimental Validation
- Run comprehensive comparisons:
  - LayerWise-QAT vs EfficientQAT
  - vs GPTQ, AWQ, OmniQuant
  - Test 2-bit, 3-bit, 4-bit quantization
  - Test different group sizes (64, 128)
- Measure: accuracy, perplexity, training time, memory usage

### Day 25-26: Ablation Studies
- Test sensitivity metric variations
- Compare static vs dynamic reordering
- Analyze per-layer sensitivity patterns
- Study sensitivity correlation with layer depth

### Day 27-28: Paper Writing & Results
- Write paper draft focusing on sensitivity analysis
- Create performance plots and sensitivity visualizations
- Document algorithm details and implementation
- Prepare code for release

---

## Technical Implementation Details

### Core Modification: `quantize/block_ap.py`
```python
# Add at the beginning of block_ap function
def block_ap(model, args, trainloader, valloader, logger=None):
    # ... existing setup code ...
    
    # NEW: Compute layer sensitivity
    if args.layer_ordering == 'sensitivity':
        logger.info("Computing layer sensitivity scores...")
        sensitivity_scores = compute_layer_sensitivity(
            model, trainloader, args.sensitivity_metric
        )
        # Reorder layers by sensitivity (most sensitive first)
        layer_indices = torch.argsort(sensitivity_scores, descending=True)
        logger.info(f"Layer sensitivity order: {layer_indices.tolist()}")
    else:
        layer_indices = list(range(len(layers)))
    
    # Modified main training loop
    for i, block_index in enumerate(layer_indices):
        logger.info(f"=== Training block {block_index} (order {i+1}/{len(layers)}) ===")
        # ... rest of existing block training code ...
```

### New File: `quantize/sensitivity_analysis.py` (~200 lines)
```python
import torch
import torch.nn as nn

def compute_layer_sensitivity(model, dataloader, metric='fisher'):
    """Compute sensitivity score for each transformer block"""
    if metric == 'fisher':
        return compute_fisher_sensitivity(model, dataloader)
    elif metric == 'gradient':
        return compute_gradient_sensitivity(model, dataloader)
    else:
        return compute_hessian_sensitivity(model, dataloader)

def compute_fisher_sensitivity(model, dataloader):
    """Fast Fisher Information diagonal approximation"""
    # Implementation here
    pass
```

### Command Line Extensions
```bash
# Add to main_block_ap.py argument parser
--layer_ordering {original,sensitivity,random}
--sensitivity_metric {fisher,gradient,hessian}
--adaptive_lr_scaling  # Scale LR based on sensitivity
```

---

## Expected Results

### Performance Improvements
- **Accuracy**: +0.3 to 0.7 points on zero-shot tasks
- **Training Speed**: 15-25% faster convergence
- **Memory**: Same or slightly lower
- **Perplexity**: 0.1-0.3 point improvement

### Key Insights for Paper
1. **Layer sensitivity patterns**: Which layers are most critical for quantization
2. **Training dynamics**: How sensitivity-based ordering affects convergence
3. **Generalization**: Sensitivity patterns across different model sizes
4. **Efficiency gains**: Quantify training speedup vs accuracy trade-off

---

## Publication Strategy

### Target: Workshop + Conference
1. **Submit to workshop first** (e.g., ICML Efficient ML Workshop)
2. **Extend for main conference** with additional analysis
3. **Focus on efficiency gains** rather than absolute accuracy

### Paper Structure
1. **Problem**: Block training order is arbitrary in current methods
2. **Method**: Sensitivity-based layer ordering for QAT
3. **Experiments**: Comprehensive evaluation on Llama-2 7B/13B
4. **Analysis**: Deep dive into sensitivity patterns and training dynamics
5. **Conclusion**: Simple but effective improvement to existing QAT

---

## Risk Mitigation

### Backup Plans
- **Week 2**: If sensitivity computation is too expensive, use simple gradient norms
- **Week 3**: If dynamic reordering fails, focus on static optimal ordering
- **Week 4**: If 13B doesn't fit, focus entirely on 7B with deeper analysis

### Validation Strategy
- **Daily**: Quick 7B validation runs
- **Weekly**: Full comparison against baselines
- **End of month**: Comprehensive evaluation ready for submission

---

## Success Criteria

### Minimum Viable Paper
- 0.2+ point accuracy improvement OR 15%+ training speedup
- Clear sensitivity analysis and layer ordering justification
- Reproducible results on Llama-2-7B

### Strong Paper
- 0.5+ point accuracy improvement AND 20%+ training speedup
- Validation on multiple model sizes (7B, 13B)
- Novel insights about layer sensitivity patterns
- Clear practical benefits for practitioners

**Timeline**: Submit to workshop by end of Week 4, extend for conference by Month 2
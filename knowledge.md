# EfficientQAT + LayerWise-QAT Knowledge

## Project Structure Understanding

### Core Training Pipeline
1. `main_block_ap.py` - Entry point for Block-AP training
2. `quantize/block_ap.py` - Core block-wise training algorithm
3. `quantize/quantizer.py` - Basic uniform quantization implementation
4. `quantize/int_linear_fake.py` - Fake quantization for training
5. `quantize/int_linear_real.py` - Real quantization for inference

### Memory Constraints (A100-40GB)
- Llama-2-7B: Safe (~8-12GB training)
- Llama-2-13B: Manageable (~15-20GB training)
- Llama-2-70B: Avoid (requires 50+ GB)

## LayerWise-QAT Implementation

### Key Modifications
1. **Added `quantize/sensitivity_analysis.py`**: Core sensitivity computation
2. **Modified `quantize/block_ap.py`**: Layer ordering and adaptive LR
3. **Extended `main_block_ap.py`**: New command-line arguments
4. **Created testing scripts**: Validation and comparison tools

### Sensitivity Metrics
- **Gradient norm**: Fastest, good for development/testing
- **Fisher Information**: Most principled, best accuracy
- **Hessian trace**: Balanced cost/accuracy trade-off

### Best Practices
- Start with gradient metric for quick iteration
- Use 32-64 samples for sensitivity computation
- Enable adaptive LR scaling for better convergence
- Always validate against original EfficientQAT baseline

### Memory Optimization Tips
- Use `--off_load_to_disk` for large datasets
- Set `--max_memory "35GiB"` to leave buffer
- Reduce batch size and use gradient accumulation
- Enable `--real_quant` to pack weights immediately

## Experimental Validation

### Testing Protocol
1. Quick validation with small datasets (64-128 samples)
2. Medium experiments with standard datasets (512-2048 samples)
3. Full validation with paper-level datasets (4096 samples)

### Expected Results
- Training speedup: 15-25%
- Accuracy improvement: 0.3-0.7 points
- Perplexity improvement: 0.1-0.3 points
- Memory usage: Same or 5-10% lower

## Publication Strategy

### Novel Contributions
1. First sensitivity-based layer ordering for QAT
2. Adaptive learning rate scaling for quantization
3. Comprehensive analysis of layer sensitivity patterns
4. Practical efficiency improvements for LLM quantization

### Target Venues
- Workshop: ICML/NeurIPS Efficient ML workshops
- Conference: ICLR, ICML, NeurIPS (with extended results)

## Development Workflow

### Phase 1 (Week 1-2): Core implementation
- Implement sensitivity analysis
- Add layer ordering to block_ap.py
- Create testing framework

### Phase 2 (Week 3): Optimization
- Test all sensitivity metrics
- Add adaptive learning rate scaling
- Optimize for A100-40GB constraints

### Phase 3 (Week 4): Validation
- Full experimental validation
- Comparison against all baselines
- Paper writing and code cleanup

## Common Issues & Solutions

### Memory Issues
- Reduce `--train_size` and `--val_size`
- Use disk offloading
- Avoid 70B models on 40GB GPU

### Training Instability
- Start with gradient metric (most stable)
- Disable adaptive LR if issues occur
- Use smaller learning rates

### Slow Sensitivity Computation
- Reduce `--sensitivity_samples`
- Use gradient metric instead of fisher/hessian
- Cache sensitivity scores between runs
#!/usr/bin/env python3
"""
Simple test script for sensitivity analysis module.
Tests the implementation without requiring a full model.
"""

import sys
import torch
import torch.nn as nn
import numpy as np

# Add current directory to path for imports
sys.path.append('.')

try:
    from quantize.sensitivity_analysis import (
        compute_layer_sensitivity,
        rank_layers_by_sensitivity, 
        analyze_sensitivity_patterns
    )
    print("✅ Successfully imported sensitivity analysis module")
except ImportError as e:
    print(f"❌ Failed to import sensitivity analysis module: {e}")
    sys.exit(1)

def create_dummy_model():
    """Create a simple dummy model for testing"""
    class DummyLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.self_attn = nn.Linear(hidden_size, hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            
        def forward(self, x, **kwargs):
            attn_out = self.self_attn(x)
            mlp_out = self.mlp(attn_out)
            return (mlp_out,)
    
    class DummyModel(nn.Module):
        def __init__(self, num_layers=4, hidden_size=512):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([DummyLayer(hidden_size) for _ in range(num_layers)])
            self.lm_head = nn.Linear(hidden_size, 1000)  # vocab size
            
        def forward(self, input_ids, labels=None, **kwargs):
            # Simple forward pass
            x = input_ids.float()  # Dummy embedding
            
            for layer in self.model.layers:
                x = layer(x)[0]
            
            logits = self.lm_head(x)
            
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1).long() % logits.size(-1)
                )
                return type('Output', (), {'loss': loss, 'logits': logits})()
            
            return type('Output', (), {'logits': logits})()
    
    return DummyModel()

def create_dummy_dataloader():
    """Create dummy data for testing"""
    # Create random sequences
    batch_size, seq_len, hidden_size = 2, 64, 512
    
    data = []
    for i in range(10):  # 10 batches
        batch = (torch.randn(batch_size, seq_len, hidden_size),)
        data.append(batch)
    
    return data

def test_sensitivity_metrics():
    """Test all sensitivity computation methods"""
    print("\n=== Testing Sensitivity Metrics ===")
    
    # Create dummy model and data
    model = create_dummy_model()
    dataloader = create_dummy_dataloader()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model layers: {len(model.model.layers)}")
    
    # Test each sensitivity metric
    metrics = ['gradient', 'fisher', 'hessian']
    results = {}
    
    for metric in metrics:
        print(f"\nTesting {metric} sensitivity...")
        try:
            start_time = time.time()
            
            # Compute sensitivity with small sample size for testing
            sensitivity_scores = compute_layer_sensitivity(
                model, dataloader, 
                metric=metric, 
                max_samples=4
            )
            
            duration = time.time() - start_time
            
            # Validate output
            assert len(sensitivity_scores) == len(model.model.layers), \
                f"Expected {len(model.model.layers)} scores, got {len(sensitivity_scores)}"
            
            assert torch.all(sensitivity_scores >= 0), "Sensitivity scores should be non-negative"
            
            # Test ranking
            layer_order = rank_layers_by_sensitivity(sensitivity_scores)
            assert len(layer_order) == len(model.model.layers), "Layer order length mismatch"
            
            # Test analysis
            analysis = analyze_sensitivity_patterns(sensitivity_scores)
            assert 'mean_sensitivity' in analysis, "Missing analysis fields"
            
            results[metric] = {
                'success': True,
                'duration': duration,
                'scores': sensitivity_scores.tolist(),
                'order': layer_order,
                'analysis': analysis
            }
            
            print(f"  ✅ {metric} PASSED ({duration:.2f}s)")
            print(f"  Sensitivity scores: {[f'{s:.3f}' for s in sensitivity_scores.tolist()]}")
            print(f"  Training order: {layer_order}")
            
        except Exception as e:
            print(f"  ❌ {metric} FAILED: {e}")
            results[metric] = {'success': False, 'error': str(e)}
    
    return results

def test_import_functionality():
    """Test basic module functionality"""
    print("=== Testing Basic Functionality ===")
    
    # Test tensor operations
    dummy_scores = torch.tensor([0.5, 1.2, 0.8, 2.1, 0.3])
    
    # Test ranking
    order_desc = rank_layers_by_sensitivity(dummy_scores, 'descending')
    order_asc = rank_layers_by_sensitivity(dummy_scores, 'ascending')
    
    print(f"Dummy scores: {dummy_scores.tolist()}")
    print(f"Descending order: {order_desc}")
    print(f"Ascending order: {order_asc}")
    
    # Verify ordering
    assert order_desc == [3, 1, 2, 0, 4], f"Expected [3,1,2,0,4], got {order_desc}"
    assert order_asc == [4, 0, 2, 1, 3], f"Expected [4,0,2,1,3], got {order_asc}"
    
    # Test analysis
    analysis = analyze_sensitivity_patterns(dummy_scores)
    print(f"Analysis: {analysis}")
    
    expected_keys = ['mean_sensitivity', 'std_sensitivity', 'max_sensitivity', 
                    'min_sensitivity', 'most_sensitive_layer', 'least_sensitive_layer']
    
    for key in expected_keys:
        assert key in analysis, f"Missing analysis key: {key}"
    
    print("✅ Basic functionality tests PASSED")
    return True

def main():
    """Run all tests"""
    print("LayerWise-QAT Sensitivity Analysis Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Basic functionality
        basic_passed = test_import_functionality()
        
        if not basic_passed:
            print("❌ Basic tests failed")
            return False
        
        # Test 2: Sensitivity metrics (requires torch)
        import time
        sensitivity_results = test_sensitivity_metrics()
        
        # Summary
        successful_metrics = sum(1 for r in sensitivity_results.values() if r['success'])
        total_metrics = len(sensitivity_results)
        
        print(f"\n=== Test Summary ===")
        print(f"Successful metrics: {successful_metrics}/{total_metrics}")
        
        if successful_metrics >= 2:  # At least 2 metrics working
            print("🎉 LayerWise-QAT sensitivity analysis is working!")
            print("\nRecommended sensitivity metric order:")
            print("1. gradient (fastest, good for development)")
            print("2. fisher (most principled, best accuracy)")
            print("3. hessian (balanced, moderate cost)")
            return True
        else:
            print("❌ Too many sensitivity metrics failed")
            return False
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
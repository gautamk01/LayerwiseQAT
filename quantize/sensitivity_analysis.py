import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import time


def compute_layer_sensitivity(model, dataloader, metric='fisher', max_samples=64):
    """
    Compute sensitivity score for each transformer block to determine training order.
    
    Args:
        model: The language model
        dataloader: Calibration data loader
        metric: Sensitivity metric to use ('fisher', 'gradient', 'hessian')
        max_samples: Maximum calibration samples to use
    
    Returns:
        torch.Tensor: Sensitivity scores for each layer
    """
    print(f"Computing sensitivity using {metric} metric with {max_samples} samples...")
    
    try:
        if metric == 'fisher':
            return compute_fisher_sensitivity(model, dataloader, max_samples)
        elif metric == 'gradient':
            return compute_gradient_sensitivity(model, dataloader, max_samples)
        elif metric == 'hessian':
            return compute_hessian_sensitivity(model, dataloader, max_samples)
        else:
            raise ValueError(f"Unknown sensitivity metric: {metric}")
    except Exception as e:
        print(f"Error in sensitivity computation: {e}")
        print("Falling back to gradient-based sensitivity...")
        return compute_gradient_sensitivity(model, dataloader, max_samples)


def compute_fisher_sensitivity(model, dataloader, max_samples=64):
    """
    Compute Fisher Information-based sensitivity scores.
    Fast diagonal approximation using single forward+backward pass.
    """
    device = next(model.parameters()).device
    layers = model.model.layers
    sensitivity_scores = torch.zeros(len(layers), device=device)
    
    # Set model to training mode for gradient computation
    model.train()
    
    # Use a dictionary to map parameters to their accumulated fisher scores
    param_to_fisher = {p: torch.zeros_like(p) for p in model.parameters()}
    
    sample_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= max_samples:
            break
            
        inputs = batch[0].to(device)
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0)
            
        # Forward pass with loss computation
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Accumulate squared gradients (Fisher Information diagonal)
        for p in model.parameters():
            if p.grad is not None:
                param_to_fisher[p] += p.grad.pow(2)
                
        sample_count += inputs.size(0)
        
        # Clear gradients
        model.zero_grad()
    
    # Compute per-layer Fisher Information
    for layer_idx, layer in enumerate(layers):
        layer_fisher = sum(param_to_fisher[p].sum() for p in layer.parameters() if p in param_to_fisher)
        sensitivity_scores[layer_idx] = layer_fisher.item()

    # Normalize scores
    if sample_count > 0:
        sensitivity_scores = sensitivity_scores / sample_count
    return sensitivity_scores.cpu()


def compute_gradient_sensitivity(model, dataloader, max_samples=64):
    """
    Compute gradient norm-based sensitivity scores.
    Fastest method - uses gradient magnitudes as proxy for sensitivity.
    """
    device = next(model.parameters()).device
    layers = model.model.layers
    sensitivity_scores = torch.zeros(len(layers), device=device)
    
    model.train()
    sample_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= max_samples:
            break
            
        inputs = batch[0].to(device)
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0)
            
        # Forward pass
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Compute gradient norms per layer
        for layer_idx, layer in enumerate(layers):
            layer_grad_norm = 0.0
            for param in layer.parameters():
                if param.grad is not None:
                    layer_grad_norm += param.grad.norm().item() ** 2
            sensitivity_scores[layer_idx] += layer_grad_norm
            
        sample_count += inputs.size(0)
        model.zero_grad()
    
    # Normalize and take square root to get actual norms
    if sample_count > 0:
        sensitivity_scores = torch.sqrt(sensitivity_scores / sample_count)
    return sensitivity_scores.cpu()


def compute_hessian_sensitivity(model, dataloader, max_samples=32):
    """
    Compute Hessian trace approximation for sensitivity.
    Medium computational cost - uses Hutchinson trace estimator.
    """
    device = next(model.parameters()).device
    layers = model.model.layers
    sensitivity_scores = torch.zeros(len(layers), device=device)
    
    model.train()
    sample_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= max_samples:
            break
            
        inputs = batch[0].to(device)
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0)
            
        try:
            # Forward pass
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            
            # First-order gradients
            first_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
            
            # Map gradients to parameters for robust indexing
            param_to_grad = {p: g for p, g in zip(model.parameters(), first_grads)}
            
            # Hutchinson trace estimator with random vectors
            for layer_idx, layer in enumerate(layers):
                layer_trace = 0.0
                
                for param in layer.parameters():
                    if param in param_to_grad:
                        grad = param_to_grad[param]
                        # Random Rademacher vector
                        z = torch.randint_like(grad, high=2, dtype=grad.dtype) * 2 - 1
                        # Hessian-vector product
                        hvp = torch.autograd.grad(grad, param, grad_outputs=z, retain_graph=True)[0]
                        # Trace contribution
                        layer_trace += (z * hvp).sum().item()
                        
                sensitivity_scores[layer_idx] += abs(layer_trace)
                
        except RuntimeError as e:
            # Handle potential autograd issues gracefully
            print(f"Warning: Hessian computation failed for batch {batch_idx}: {e}")
            continue
            
        sample_count += inputs.size(0)
    
    # Normalize scores
    if sample_count > 0:
        sensitivity_scores = sensitivity_scores / sample_count
    return sensitivity_scores.cpu()


def rank_layers_by_sensitivity(sensitivity_scores, order='descending'):
    """
    Rank layers by sensitivity scores.
    
    Args:
        sensitivity_scores: Tensor of sensitivity scores
        order: 'descending' (most sensitive first) or 'ascending'
    
    Returns:
        List of layer indices in training order
    """
    if order == 'descending':
        return torch.argsort(sensitivity_scores, descending=True).tolist()
    else:
        return torch.argsort(sensitivity_scores, descending=False).tolist()


def analyze_sensitivity_patterns(sensitivity_scores, layer_names=None):
    """
    Analyze and visualize sensitivity patterns across layers.
    
    Returns:
        Dict with analysis results
    """
    analysis = {
        'mean_sensitivity': sensitivity_scores.mean().item(),
        'std_sensitivity': sensitivity_scores.std().item(),
        'max_sensitivity': sensitivity_scores.max().item(),
        'min_sensitivity': sensitivity_scores.min().item(),
        'sensitivity_range': (sensitivity_scores.max() - sensitivity_scores.min()).item(),
        'most_sensitive_layer': sensitivity_scores.argmax().item(),
        'least_sensitive_layer': sensitivity_scores.argmin().item()
    }
    
    return analysis
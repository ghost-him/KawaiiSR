import argparse
import os
import sys
import yaml
import torch
import torch.onnx
import math

# Add the directory containing the model definitions to path
sys.path.append(os.path.join(os.getcwd(), 'models'))

from KawaiiSR.KawaiiSR import KawaiiSR
from KawaiiSR.MambaIRv2 import MambaIRv2
import torch.nn.functional as F
from typing import Optional

# Define the scripted function for ONNX loop support
@torch.jit.script
def _selective_scan_loop(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: Optional[torch.Tensor], delta_bias: Optional[torch.Tensor], delta_softplus: bool) -> torch.Tensor:
    b_sz, d_sz, l_sz = u.shape
    n_sz = A.shape[1]
    
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
        
    if delta_softplus:
        delta = F.softplus(delta)
        
    h = torch.zeros((b_sz, d_sz, n_sz), device=u.device, dtype=u.dtype)
    
    ys = torch.jit.annotate(list[torch.Tensor], [])
    
    # Debug print
    # print("DEBUG JIT u:", u.shape)
    
    for i in range(l_sz):
        u_t = u[:, :, i]
        delta_t = delta[:, :, i]
        B_t = B[:, :, i]
        C_t = C[:, :, i]
        
        # bar_A = exp(delta_t * A)
        bar_A = torch.exp((delta_t.unsqueeze(-1) * A).float()).to(u.dtype)
        # bar_B = delta_t * B_t
        bar_B = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
        
        # print("DEBUG Loop:", h.shape, bar_A.shape, bar_B.shape, u_t.shape)

        h = bar_A * h + bar_B * u_t.unsqueeze(-1)
        
        y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)
        
        if D is not None:
             y_t = y_t + u_t * D
             
        ys.append(y_t)
        
    return torch.stack(ys, dim=-1)

def selective_scan_onnx(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    # Handle dimensionality mismatch between u (B, D, L) and B (B, K, N, L)
    b_sz = u.shape[0]
    was_reshaped = False
    original_3d = False
    k_sz = 1
    d_head = 0
    l_sz = 0

    if u.dim() == 3 and B.dim() == 4:
        # MambaIRv2 case: u flattens K, B keeps K
        b_sz, d_total, l_sz = u.shape
        b_b, k_sz, n_sz, l_b = B.shape
        
        d_head = d_total // k_sz
        
        # Reshape u, delta to (B, K, D_head, L) and then flatten B*K
        u = u.view(b_sz, k_sz, d_head, l_sz).flatten(0, 1)
        delta = delta.view(b_sz, k_sz, d_head, l_sz).flatten(0, 1)
        B = B.flatten(0, 1)
        C = C.flatten(0, 1)
        was_reshaped = True
        original_3d = True
        
    elif u.dim() == 4:
        # u is (B, K, D, L)
        b, k_sz, d_head, l = u.shape
        b_sz = b
        l_sz = l
        u = u.flatten(0, 1)
        delta = delta.flatten(0, 1)
        B = B.flatten(0, 1)
        C = C.flatten(0, 1)
        was_reshaped = True
        original_3d = False
        
    y = _selective_scan_loop(u, delta, A, B, C, D, delta_bias, delta_softplus)
    
    if was_reshaped:
        # y is (B*K, D_head, L) -> Reshape to original u shape
        if original_3d:
             # Original u dim 3: (B, D_total, L)
             y = y.view(b_sz, k_sz, d_head, y.shape[-1]).flatten(1, 2)
        else:
             # Original u dim 4
             y = y.view(b_sz, k_sz, d_head, y.shape[-1])
             
    if z is not None:
        y = y * F.silu(z)
        
    if return_last_state:
        return y, None
    return y

import KawaiiSR.MambaIRv2 as MambaIRv2Mod
MambaIRv2Mod.selective_scan_fn = selective_scan_onnx

# Patch gumbel_softmax to be deterministic (remove random sampling) for ONNX export
# This avoids aten::exponential error
def deterministic_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # Fix: Include tau scaling
    y_soft = (logits / tau).softmax(dim)
    
    if hard:
        # Straight-through estimator for argmax (deterministic)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # For inference ONNX export, we can just return y_hard
        ret = y_hard
    else:
        ret = y_soft
    return ret


F.gumbel_softmax = deterministic_gumbel_softmax

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'model_config' in config:
        return config['model_config']
    elif 'model' in config:
        return config['model']
    else:
        # Fallback or assume simple structure
        return config

def export_onnx(config_path, tile_size, padding, output_path):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Using device: {device}")
    
    input_size = tile_size
    effective_size = input_size - 2 * padding
    
    print(f"Configuration:")
    print(f"  Input Size (ONNX): {input_size}x{input_size} (User Requested)")
    print(f"  Padding (Edge):    {padding} (Approx 10% of tile size, aligned to window_size 16)")
    print(f"  Effective Size:    {effective_size}x{effective_size}")
    
    # Use a small fixed size for tracing to avoid OOM
    trace_size = 128
    
    model_config = load_config(config_path)
    
    # Extract parameters
    upscale = model_config['upscale']
    embed_dim = model_config['embed_dim']
    d_state = model_config['d_state']
    depths = model_config['depths']
    num_heads = model_config['num_heads']
    window_size = model_config['window_size']
    inner_rank = model_config['inner_rank']
    num_tokens = model_config['num_tokens']
    convffn_kernel_size = model_config['convffn_kernel_size']
    mlp_ratio = model_config['mlp_ratio']
    upsampler = model_config['upsampler']
    resi_connection = model_config['resi_connection']
    image_range = model_config['image_range']
    
    print("Initializing KawaiiSR Model...")
    # Instantiate KawaiiSR instead of MambaIRv2 directly
    model = KawaiiSR(
        image_size=trace_size, 
        in_channels=3,
        image_range=image_range,
        # MambaIR args
        upscale=upscale,
        embed_dim=embed_dim,
        d_state=d_state,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        inner_rank=inner_rank,
        num_tokens=num_tokens,
        convffn_kernel_size=convffn_kernel_size,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=resi_connection
    ).to(device)
    model.eval()
    
    # Dummy input
    # Use a small fixed size for tracing to avoid OOM, while allowing dynamic shapes in the output model
    print(f"Tracing with dummy input size: {trace_size}x{trace_size} to prevent OOM...")
    dummy_input = torch.randn(1, 3, trace_size, trace_size, device=device)
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to {output_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=20,
            dynamo=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {2: 'height', 3: 'width'},
                'output': {2: 'height', 3: 'width'}
            }
        )
        print("Export successful!")
    except Exception as e:
        print(f"Export failed: {e}")
        print("Try reducing --tile_size if OOM occurs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export KawaiiSR (MambaIR) to ONNX")
    parser.add_argument("--config", type=str, default="models/configs/real_stage1.yaml", help="Path to config file")
    # Updated default to 512 as requested
    parser.add_argument("--tile_size", type=int, default=256, help="Input size for the ONNX model (Total: Effective + 2*Padding)")
    # Updated default to 48 (approx 10% of 512, 16-aligned)
    parser.add_argument("--padding", type=int, default=48, help="Padding size (Edge)")
    parser.add_argument("--output", type=str, default="models/kawaii_sr.onnx", help="Output ONNX file path")
    
    args = parser.parse_args()
    
    export_onnx(args.config, args.tile_size, args.padding, args.output)

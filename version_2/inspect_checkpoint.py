"""
Week 2-3: LSTM Checkpoint Loader - Smart Architecture Detection
Automatically detect model architecture from checkpoint and load correctly.
"""

import torch
import json
from pathlib import Path

def inspect_checkpoint_architecture(checkpoint_path: Path) -> dict:
    """Inspect checkpoint to determine model architecture"""
    
    print(f"\nInspecting checkpoint: {checkpoint_path}")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Print all keys and shapes
        print("\nCheckpoint state dict keys and shapes:")
        print("="*80)
        
        architecture_info = {}
        
        for key, tensor in state_dict.items():
            print(f"{key:60} {list(tensor.shape)}")
            
            # Extract architecture info
            if 'input_projection.weight' in key:
                d_model, input_dim = tensor.shape
                architecture_info['d_model'] = d_model
                architecture_info['input_dim'] = input_dim
            
            if 'temporal_encoder.transformer_encoder.layers.0' in key:
                if 'self_attn.in_proj_weight' in key:
                    # This should be (3*d_model, d_model)
                    total_proj, d_model = tensor.shape
                    if architecture_info.get('d_model') is None:
                        architecture_info['d_model'] = d_model
        
        return architecture_info
        
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        return None

def main():
    checkpoint_path = Path('e:/icu_project/checkpoints/multimodal/fold_0_best_model.pt')
    
    if checkpoint_path.exists():
        info = inspect_checkpoint_architecture(checkpoint_path)
        print("\n" + "="*80)
        print("DETECTED ARCHITECTURE:")
        print("="*80)
        if info:
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # Save architecture info
        output_file = Path('e:/icu_project/checkpoint_architecture.json')
        if info:
            with open(output_file, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"\n✓ Architecture saved to {output_file}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

if __name__ == '__main__':
    main()

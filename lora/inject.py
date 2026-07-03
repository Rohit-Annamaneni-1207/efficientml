import torch
import torch.nn as nn
from .layers import LoRALayer

def inject_lora(model, rank, alpha, target_modules):
    
    for name, module in model.named_modules():
        if not any(t in name for t in target_modules) or not isinstance(module, nn.Linear):
            continue

        parts = name.split(".")

        parent_path = parts[:-1]
        child = parts[-1]

        parent = model
        for part in parent_path:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        setattr(parent, child, LoRALayer(module, rank, alpha))

    return model


        

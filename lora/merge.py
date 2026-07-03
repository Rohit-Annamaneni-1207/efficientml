import torch
import torch.nn as nn
from .layers import LoRALayer

def merge_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            base_linear = module.linear
            delta_weight = module.B.weight @ module.A.weight

            merged_weight = base_linear.weight + delta_weight

            merged_linear = nn.Linear(
                base_linear.in_features, 
                base_linear.out_features, 
                bias=base_linear.bias is not None
            )

            merged_linear.weight.data.copy_(merged_weight)

            if base_linear.bias is not None:
                merged_linear.bias.data.copy_(base_linear.bias.data)

            # Find parent module

            parts = name.split(".")
            parent_path = parts[:-1]
            child_name = parts[-1]
            parent = model

            for part in parent_path:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)

            # Replace LoRALayer with merged nn.Linear
            setattr(parent, child_name, merged_linear)

    return model



            

        
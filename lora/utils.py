from .layers import LoRALayer

def mark_only_lora_trainable(model):

    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALayer):
            for param in module.A.parameters():
                param.requires_grad = True

            for param in module.B.parameters():
                param.requires_grad = True

    return model
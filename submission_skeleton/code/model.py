
import timm
import torch
from torch import nn

def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Create a timm model with given num_classes.
    Defaults to widely available backbones. Examples:
      - 'efficientnet_b0'
      - 'convnext_tiny'
      - 'swin_tiny_patch4_window7_224'
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def load_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = True, map_location='cpu'):
    state = torch.load(ckpt_path, map_location=map_location)
    # allow state dict or full dict
    sd = state.get('state_dict', state)
    # strip "module." if present
    new_sd = {}
    for k,v in sd.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=strict)
    return model

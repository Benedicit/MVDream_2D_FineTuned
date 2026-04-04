# mvdream/ldm/modules/lora.py
import torch
from torch import nn

from typing import Iterable, Tuple
from mvdream.ldm.modules.attention import CrossAttention, MemoryEfficientCrossAttention, SpatialSelfAttention


class LoRALinear(nn.Module):
    """
    Wrap a Linear layer with a low-rank LoRA adapter.

    This is deliberately minimal: no bias, no fancy init beyond zeroing
    the LoRA-up layer (so you start exactly from the base model).
    """
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha

        self.lora_down = nn.Linear(base.in_features, r, bias=False, device=base.weight.device)
        self.lora_up   = nn.Linear(r, base.out_features, bias=False, device=base.weight.device)


        nn.init.zeros_(self.lora_up.weight)
        nn.init.normal_(self.lora_down.weight, std=1e-4)

        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale

class LoRAConv2d(nn.Module):
    """
    Wrap a Conv2d layer with a low-rank LoRA adapter.
    """
    def __init__(self, base: nn.Conv2d, r: int = 4, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        self.r = r
        self.alpha = alpha

        # down-layer uses base kernel/stride/padding, up-layer is 1x1
        self.lora_down = nn.Conv2d(
            base.in_channels,
            r,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            device=base.weight.device,
            bias=False
        )
        self.lora_up = nn.Conv2d(r, base.out_channels, kernel_size=1, stride=1, padding=0, device=base.weight.device, bias=False)

        nn.init.zeros_(self.lora_up.weight)
        nn.init.normal_(self.lora_down.weight, std=1e-4)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale

LORA_TARGET_DEFAULT = ("to_q", "to_k", "to_v", "to_out.0")

def _wrap_linear_with_lora(module: nn.Module, attr: str, r: int, alpha: float):
    parts = attr.split(".")
    sub = module
    for p in parts[:-1]:
        sub = getattr(sub, p)
    last_name = parts[-1]
    base_layer = getattr(sub, last_name)
    wrapped = LoRALinear(base_layer, r=r, alpha=alpha)
    setattr(sub, last_name, wrapped)

def add_lora_to_cross_att_only(
    unet: nn.Module,
    r: int = 4,
    alpha: float = 1.0,
    target_linear_names: Iterable[str] = LORA_TARGET_DEFAULT,
) -> Tuple[int, int]:
    """
    Walk the MultiViewUNetModel and wrap its attention linears with LoRA.

    Returns (num_attn_modules, num_lora_layers) for sanity checking.
    """
    num_attn = 0
    num_lora = 0

    for module in unet.modules():
        if isinstance(module, (CrossAttention, MemoryEfficientCrossAttention)):
            num_attn += 1
            for name in target_linear_names:
                _wrap_linear_with_lora(module, name, r=r, alpha=alpha)
                num_lora += 1

    return num_attn, num_lora

def add_lora_to_attention_and_conv(model: nn.Module, r: int = 4, alpha: float = 1.0):
    """
    Add LoRA layers to attention and convolution layers in the model.
    """
    for name, module in model.named_children():

        # 1. If it's a Convolution, wrap it directly
        if isinstance(module, nn.Conv2d):
            wrapped = LoRAConv2d(module, r=r, alpha=alpha)
            setattr(model, name, wrapped)

        # 2. If it's an Attention module, go inside and wrap Linears (and Convs if any)
        elif isinstance(module, (CrossAttention, MemoryEfficientCrossAttention, SpatialSelfAttention)):
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear):
                    wrapped = LoRALinear(sub_module, r=r, alpha=alpha)
                    setattr(module, sub_name, wrapped)
                elif isinstance(sub_module, nn.Conv2d):
                    wrapped = LoRAConv2d(sub_module, r=r, alpha=alpha)
                    setattr(module, sub_name, wrapped)
                else:
                    add_lora_to_attention_and_conv(sub_module, r=r, alpha=alpha)

        # 4. For generic containers (Sequential, ModuleList, Block, etc.), just recurse.
        else:
            add_lora_to_attention_and_conv(module, r=r, alpha=alpha)
def add_lora_to_all_layers(model: nn.Module, r: int = 64, alpha: float = 1.0):
    """
    Recursively add LoRA layers to all nn.Linear and nn.Conv2d layers in the model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            wrapped = LoRALinear(module, r=r, alpha=alpha)
            setattr(model, name, wrapped)
        elif isinstance(module, nn.Conv2d):
            wrapped = LoRAConv2d(module, r=r, alpha=alpha)
            setattr(model, name, wrapped)
        else:
            # Recurse into custom modules, Sequential, ModuleList, etc.
            add_lora_to_all_layers(module, r=r, alpha=alpha)
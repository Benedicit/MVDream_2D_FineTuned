# mvdream/ldm/modules/lora.py
import torch
from torch import nn

from typing import Iterable, Tuple
from mvdream.ldm.modules.attention import CrossAttention, MemoryEfficientCrossAttention

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

        self.lora_down = nn.Linear(base.in_features, r, bias=False)
        self.lora_up   = nn.Linear(r, base.out_features, bias=False)

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

def add_lora_to_mvdream_unet(
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

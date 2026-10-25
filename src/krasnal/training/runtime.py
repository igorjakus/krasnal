"""Shared training pieces."""

import math
from dataclasses import asdict
from pathlib import Path

import torch


def cosine_lr(step: int, warmup: int, steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


class EMA:
    """Exponential moving average of model weights, evaluated as a shadow model."""

    def __init__(self, model, decay: float = 0.999, every: int = 10):
        self.decay = decay
        self.every = every
        self.shadow = {
            k.replace("_orig_mod.", ""): v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model, step: int):
        if step % self.every:
            return
        state = model.state_dict()
        for k, v in state.items():
            # torch.compile prefixes keys with _orig_mod.; strip it so the shadow
            # matches both the raw and the compiled module.
            shadow = self.shadow[k.replace("_orig_mod.", "")]
            if v.dtype.is_floating_point:
                shadow.lerp_(v.detach().float(), 1 - self.decay)
            else:
                shadow.copy_(v)

    @torch.no_grad()
    def copy_to(self, model):
        """Write EMA weights into a model instance for evaluation."""
        state = model.state_dict()
        for k, v in state.items():
            target = self.shadow.get(k.replace("_orig_mod.", ""))
            if target is None:
                continue
            v.copy_(target.to(v.dtype))
        return model


def save(path: Path, model, step: int, ema: EMA | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": state, "config": asdict(model.config), "step": step}, path)
    if ema is not None:
        ema_path = path.parent / "model-ema.pt"
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        ema_state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
        torch.save({"model": ema_state, "config": asdict(model.config), "step": step}, ema_path)
        model.load_state_dict(backup)


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"

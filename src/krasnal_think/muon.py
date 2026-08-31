"""Muon optimizer: orthogonalized momentum for 2D parameters.

Keller Jordan's reference implementation: the update is Newton-Schulz-
orthogonalized momentum, scaled so its RMS matches the weight's sqrt(shape)
scaling rule. Embeddings, the tied lm_head, and any <2D parameter stay on
AdamW.
"""

import torch


@torch.compile
def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonalization UV^T of G via Newton-Schulz (quintic)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                update = _newton_schulz(update)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update.reshape(p.shape), alpha=-group["lr"] * scale)


def split_params(model, weight_decay: float):
    """Muon gets 2D non-embedding matrices; AdamW gets the rest."""
    muon, adamw_decay, adamw_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_2d_matrix = p.dim() >= 2 and "wte" not in name and "lm_head" not in name
        if is_2d_matrix:
            muon.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)
    groups = [
        {"params": muon, "use_muon": True, "lr": None, "weight_decay": 0.0},
        {"params": adamw_decay, "use_muon": False, "lr": None, "weight_decay": weight_decay},
        {"params": adamw_nodecay, "use_muon": False, "lr": None, "weight_decay": 0.0},
    ]
    return groups


class OptimizerPair:
    """Stepped together; per-group LR set externally each step."""

    def __init__(self, model, weight_decay: float, muon_lr: float, adamw_lr: float):
        groups = split_params(model, weight_decay)
        self.muon = Muon([dict(g, lr=muon_lr) for g in groups if g["use_muon"]])
        self.adamw = torch.optim.AdamW(
            [dict(g, lr=adamw_lr) for g in groups if not g["use_muon"]],
            betas=(0.9, 0.95),
            fused=False,
        )

    def set_lr(self, muon_lr: float, adamw_lr: float):
        for g in self.muon.param_groups:
            g["lr"] = muon_lr
        for g in self.adamw.param_groups:
            g["lr"] = adamw_lr

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

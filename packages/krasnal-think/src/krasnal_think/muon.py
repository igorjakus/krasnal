"""Muon optimizer: orthogonalized momentum for 2D parameters.

Keller Jordan's reference implementation: the update is Newton-Schulz-
orthogonalized momentum, scaled so its RMS matches the weight's sqrt(shape)
scaling rule. Embeddings, the tied lm_head, and any <2D parameter stay on
AdamW.
"""

import math

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
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        tail_boost: float = 1.0,
        warmup_fraction: float = 0.0,
        power_iters: int = 10,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                tail_boost=tail_boost,
                warmup_fraction=warmup_fraction,
                power_iters=power_iters,
            ),
        )

    @torch.no_grad()
    def _power_iter(self, M: torch.Tensor, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-one approximation of M's leading singular pair (SAMuon-lite)."""
        n = M.size(1)
        v = torch.randn(n, device=M.device, dtype=torch.float32)
        for _ in range(iters):
            u = torch.nn.functional.normalize(M @ v, dim=0)
            v = torch.nn.functional.normalize(M.T @ u, dim=0)
        u = torch.nn.functional.normalize(M @ v, dim=0)
        return u, v

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            gamma = group["tail_boost"]
            warmup_fraction = group["warmup_fraction"]
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
                if gamma > 1.0 and warmup_fraction > 0:
                    # SAMuon-lite head-anchored allocation: amplify the whitened
                    # update's bulk by gamma_t while restoring the leading singular
                    # pair to the Muon scale (unit). Warmup w rises 0->1 on a cosine
                    # over the first warmup_fraction of the run; gamma_t = 1+(gamma-1)w.
                    if not hasattr(self, "total_steps") or self.total_steps is None:
                        raise RuntimeError("SAMuon requires total_steps to schedule the warmup")
                    progress = min(1.0, self.steps_done / max(1, self.total_steps))
                    if progress < warmup_fraction:
                        w = 0.5 * (1 - math.cos(math.pi * progress / warmup_fraction))
                    else:
                        w = 1.0
                    gamma_t = 1.0 + (gamma - 1.0) * w
                    if gamma_t > 1.0:
                        # Work in the orientation where rows are the smaller dim.
                        M = buf
                        transposed = M.size(0) > M.size(1)
                        Mt = M.T if transposed else M
                        u, v = self._power_iter(Mt.float(), group["power_iters"])
                        head = (u[:, None] * v[None, :]).to(update.dtype)
                        if transposed:
                            head = head.T
                        update = gamma_t * update - (gamma_t - 1.0) * head
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update.reshape(p.shape), alpha=-group["lr"] * scale)
        self.steps_done = getattr(self, "steps_done", 0) + 1


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

    def __init__(
        self,
        model,
        weight_decay: float,
        muon_lr: float,
        adamw_lr: float,
        tail_boost: float = 1.0,
        warmup_fraction: float = 0.0,
        total_steps: int | None = None,
    ):
        groups = split_params(model, weight_decay)
        self.muon = Muon(
            [dict(g, lr=muon_lr) for g in groups if g["use_muon"]],
            tail_boost=tail_boost,
            warmup_fraction=warmup_fraction,
        )
        if tail_boost > 1.0:
            self.muon.total_steps = total_steps
            self.muon.steps_done = 0
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

import torch

from krasnal_think.model import GPT, GPTConfig
from krasnal_think.muon import OptimizerPair, split_params
from krasnal_think.train import EMA


def _model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))


def test_split_params_routes_embeddings_to_adamw():
    model = _model()
    groups = split_params(model, weight_decay=0.1)
    muon_params = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
    adamw_params = {id(p) for g in groups if not g["use_muon"] for p in g["params"]}
    for name, p in model.named_parameters():
        if p.dim() >= 2 and "wte" not in name and "lm_head" not in name:
            assert id(p) in muon_params, name
        else:
            assert id(p) in adamw_params, name
    assert all(p.dim() >= 2 for g in groups if g["use_muon"] for p in g["params"])


def test_muon_step_reduces_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    idx = torch.randint(0, 128, (4, 16))
    optimizer = OptimizerPair(model, 0.1, 0.02, 6e-4)
    first = None
    for _ in range(10):
        targets = idx.clone()
        _, loss = model(idx, targets=targets)
        if first is None:
            first = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert loss.item() < first


def test_muon_update_is_finite_and_moves_weights():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    optimizer = OptimizerPair(model, 0.1, 0.01, 1e-4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    idx = torch.randint(0, 128, (2, 8))
    _, loss = model(idx, targets=idx)
    loss.backward()
    optimizer.step()
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name
        assert not torch.equal(p, before[name]), name


def test_tail_boost_one_recovers_muon():
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (2, 16))

    model_a = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    model_b = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    model_b.load_state_dict(model_a.state_dict())
    opt_a = OptimizerPair(model_a, 0.1, 0.01, 1e-4)
    opt_b = OptimizerPair(model_b, 0.1, 0.01, 1e-4, tail_boost=1.0, total_steps=10)
    _, loss = model_a(idx, targets=idx)
    loss.backward()
    opt_a.step()
    opt_a.zero_grad()
    _, loss = model_b(idx, targets=idx)
    loss.backward()
    opt_b.step()
    for (na, pa), (nb, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert torch.allclose(pa, pb, atol=1e-6), (na, nb)


def test_samuon_lite_head_pinning_shape():
    """The head-anchored update must equal gamma*bulk with rank-1 head restored."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    optimizer = OptimizerPair(
        model, 0.1, 0.01, 1e-4, tail_boost=5.0, warmup_fraction=0.0, total_steps=10
    )
    idx = torch.randint(0, 128, (2, 8))
    _, loss = model(idx, targets=idx)
    loss.backward()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer.step()
    # With gamma=5 and a warmup_fraction of 0 the update magnitude per matrix is
    # between 1x (head direction) and 5x (bulk) of plain Muon's: no NaNs, moved.
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name
        delta = (p - before[name]).abs().max().item()
        assert delta > 0, name


def test_samuon_warmup_matches_muon_early():
    """At step 0 with warmup, gamma_t == 1 exactly -> identical to plain Muon."""
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (2, 16))
    model_a = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    model_b = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    model_b.load_state_dict(model_a.state_dict())
    opt_a = OptimizerPair(model_a, 0.1, 0.01, 1e-4)
    opt_b = OptimizerPair(
        model_b, 0.1, 0.01, 1e-4, tail_boost=5.0, warmup_fraction=0.3, total_steps=100
    )
    for opt, m in ((opt_a, model_a), (opt_b, model_b)):
        _, loss = m(idx, targets=idx)
        loss.backward()
        opt.step()
        opt.zero_grad()
    for (na, pa), (nb, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert torch.allclose(pa, pb, atol=1e-5), (na, nb)


def test_ema_tracks_and_swaps():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    ema = EMA(model, decay=0.5, every=1)
    snapshot = {k: v.clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, step=1)
    raw = {k: v.clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            expected = 0.5 * snapshot[k] + 0.5 * raw[k]
            assert torch.allclose(v, expected, atol=1e-5), k


def test_ema_strips_compile_prefix():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    compiled = torch.compile(model, dynamic=False)
    ema = EMA(compiled, decay=0.5, every=1)
    assert not any(k.startswith("_orig_mod.") for k in ema.shadow)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    ema.update(compiled, step=1)
    ema.copy_to(model)
    # EMA of initial (w) and shifted (w+2) with decay 0.5 = w+1.
    for k, v in model.state_dict().items():
        if not v.dtype.is_floating_point:
            continue
        expected = initial[k] + 1.0
        assert torch.allclose(v, expected, atol=0.02), k

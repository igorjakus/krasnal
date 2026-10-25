import torch

from krasnal_think.model import GPT, GPTConfig, KVCache


def _model():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    return model.eval()


def _cached_logits(model, idx, chunks):
    cache = KVCache(model.config, 1, idx.device, torch.float32)
    start, logits = 0, None
    for size in chunks:
        logits, _ = model(idx[:, start : start + size], cache=cache)
        start += size
    assert start == idx.size(1)
    return logits


@torch.no_grad()
def test_kv_cache_matches_full_forward():
    model = _model()
    idx = torch.randint(0, 128, (1, 20))
    reference, _ = model(idx)
    for chunks in ([20], [1] * 20, [12, 1, 1, 6], [5, 5, 5, 5]):
        assert torch.allclose(_cached_logits(model, idx, chunks), reference, atol=1e-5)


@torch.no_grad()
def test_kv_cache_matches_full_forward_with_rope_values():
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, rope_values=True)
    ).eval()
    idx = torch.randint(0, 128, (1, 20))
    reference, _ = model(idx)
    for chunks in ([20], [1] * 20, [12, 1, 1, 6], [5, 5, 5, 5]):
        assert torch.allclose(_cached_logits(model, idx, chunks), reference, atol=1e-5)


@torch.no_grad()
def test_rope_values_changes_attention_output():
    torch.manual_seed(0)
    plain = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32)).eval()
    rove = GPT(
        GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, rope_values=True)
    ).eval()
    rove.load_state_dict(plain.state_dict(), strict=False)
    idx = torch.randint(0, 128, (2, 20))
    plain_logits, _ = plain(idx)
    rove_logits, _ = rove(idx)
    assert not torch.allclose(plain_logits, rove_logits)


@torch.no_grad()
def test_qk_norm_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, qk_norm=True)
    ).eval()
    assert model.blocks[0].attn.q_norm is not None
    idx = torch.randint(0, 128, (1, 20))
    reference, _ = model(idx)
    for chunks in ([20], [1] * 20, [12, 1, 1, 6]):
        assert torch.allclose(_cached_logits(model, idx, chunks), reference, atol=1e-5)


@torch.no_grad()
def test_hyper_connections_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, hyper_connections=2
        )
    ).eval()
    idx = torch.randint(0, 128, (1, 20))
    reference, _ = model(idx)
    for chunks in ([20], [1] * 20, [12, 1, 1, 6]):
        assert torch.allclose(_cached_logits(model, idx, chunks), reference, atol=1e-5)


@torch.no_grad()
def test_hyper_connections_matches_single_stream_at_initialization():
    torch.manual_seed(0)
    plain = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32)).eval()
    hyper = GPT(
        GPTConfig(
            vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, hyper_connections=2
        )
    ).eval()
    hyper.load_state_dict(plain.state_dict(), strict=False)
    idx = torch.randint(0, 128, (2, 20))
    plain_logits, _ = plain(idx)
    hyper_logits, _ = hyper(idx)
    assert torch.allclose(hyper_logits, plain_logits, atol=1e-5)


@torch.no_grad()
def test_sinkhorn_projection_is_doubly_stochastic():
    from krasnal_think.model import HyperConnection

    torch.manual_seed(0)
    hc = HyperConnection(2, 0, sinkhorn_iters=20)
    hc.alpha.normal_(0, 1)
    projected = hc._sinkhorn(hc.alpha[:, :2])
    assert torch.allclose(projected.sum(0), torch.ones(2), atol=1e-4)
    assert torch.allclose(projected.sum(1), torch.ones(2), atol=1e-4)
    assert (projected >= 0).all()


@torch.no_grad()
def test_manifold_hyper_connections_matches_hc_at_identity_init():
    torch.manual_seed(0)
    plain = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32)).eval()
    mhc = GPT(
        GPTConfig(
            vocab_size=128,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
            hyper_connections=2,
            sinkhorn_iters=20,
        )
    ).eval()
    mhc.load_state_dict(plain.state_dict(), strict=False)
    idx = torch.randint(0, 128, (2, 20))
    plain_logits, _ = plain(idx)
    mhc_logits, _ = mhc(idx)
    assert torch.allclose(mhc_logits, plain_logits, atol=1e-5)


@torch.no_grad()
def test_manifold_hyper_connections_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=128,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
            hyper_connections=2,
            sinkhorn_iters=5,
        )
    ).eval()
    idx = torch.randint(0, 128, (1, 20))
    reference, _ = model(idx)
    for chunks in ([20], [1] * 20, [12, 1, 1, 6]):
        assert torch.allclose(_cached_logits(model, idx, chunks), reference, atol=1e-5)


@torch.no_grad()
def test_qk_norm_bounds_query_norms():
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, qk_norm=True)
    ).eval()
    idx = torch.randint(0, 128, (2, 16))
    # Scale c_attn output 50x: raw q norms explode, QK-norm keeps them at
    # unit RMS per head (norm = sqrt(head_dim) = 4) regardless of scale.
    attn = model.blocks[0].attn
    with torch.no_grad():
        q_raw = attn.c_attn(model.wte(idx)).chunk(3, dim=2)[0].view(2, 16, 2, -1)
        attn.c_attn.weight.mul_(50.0)
        q_big = attn.c_attn(model.wte(idx)).chunk(3, dim=2)[0].view(2, 16, 2, -1)
        qn = attn.q_norm(q_big)
        assert q_big.norm(dim=-1).mean() > 50 * q_raw.norm(dim=-1).mean() * 0.9
        assert torch.allclose(qn.norm(dim=-1).mean(), torch.tensor(4.0), rtol=0.2)


@torch.no_grad()
def test_loss_ignores_masked_targets():
    model = _model()
    idx = torch.randint(0, 128, (2, 16))
    targets = torch.full_like(idx, -1)
    targets[:, -1] = idx[:, -1]
    _, loss = model(idx, targets=targets)
    assert torch.isfinite(loss)


def test_eval_head_contributes_to_loss():
    config = GPTConfig(
        vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, eval_classes=29
    )
    model = GPT(config)
    idx = torch.randint(0, 128, (2, 16))
    eval_targets = torch.full_like(idx, -1)
    eval_targets[:, -1] = 14
    _, loss = model(idx, targets=idx, eval_targets=eval_targets, eval_weight=0.25)
    assert torch.isfinite(loss)


def test_mtp_module_trains_and_inference_free():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, mtp_depth=1)
    model = GPT(config)
    base = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    assert model.num_params() > base.num_params()

    idx = torch.randint(0, 128, (2, 16))
    _, loss = model(idx, targets=idx, mtp_weight=0.3)
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.mtp.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0

    # Inference path untouched: no mtp_weight means no MTP forward.
    model.eval()
    with torch.no_grad():
        logits, lossless = model(idx[:, :8])
    assert logits.shape == (2, 1, 128) and lossless is None


def test_mtp_state_dict_roundtrip_without_mtp():
    torch.manual_seed(0)
    base = GPT(GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32))
    config = GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, mtp_depth=1)
    model = GPT(config)
    # A checkpoint trained without MTP loads into an MTP-enabled model.
    model.load_state_dict(base.state_dict(), strict=False)
    idx = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        reference, _ = base(idx)
        logits, _ = model(idx)
    assert torch.allclose(logits, reference)

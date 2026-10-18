import pytest
import torch

from model.transformer import CausalSelfAttention, MiniLM
from tests.helpers import tiny_model_cfg


def test_forward_pass_shapes_and_loss():
    cfg = tiny_model_cfg(vocab_size=20)
    model = MiniLM(cfg)
    x = torch.randint(0, 20, (4, 16))
    y = torch.randint(0, 20, (4, 16))
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, 20)
    assert loss is not None and loss.ndim == 0
    logits_only, no_loss = model(x)
    assert no_loss is None and logits_only.shape == (4, 16, 20)


def test_initial_loss_is_close_to_uniform_guessing():
    import math

    cfg = tiny_model_cfg(vocab_size=30)
    model = MiniLM(cfg).eval()
    x = torch.randint(0, 30, (8, 16))
    _, loss = model(x, x)
    assert abs(loss.item() - math.log(30)) < 0.5


def test_sequence_longer_than_context_is_rejected():
    model = MiniLM(tiny_model_cfg(vocab_size=10))
    with pytest.raises(ValueError):
        model(torch.randint(0, 10, (1, 17)))


def test_attention_weights_are_causal_and_normalised():
    cfg = tiny_model_cfg(vocab_size=10)
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 10, cfg.n_embd)
    _, weights = attn(x, return_attention=True)          # (B, heads, T, T)
    assert weights.shape == (2, cfg.n_head, 10, 10)
    upper = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[:, :, upper] == 0), "attention to future positions must be exactly zero"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, cfg.n_head, 10), atol=1e-5)


def test_changing_future_tokens_does_not_change_past_logits():
    cfg = tiny_model_cfg(vocab_size=25)
    model = MiniLM(cfg).eval()
    a = torch.randint(0, 25, (1, 12))
    b = a.clone()
    b[:, 6:] = (b[:, 6:] + 7) % 25                       # alter only positions 6..11
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :6], lb[:, :6], atol=1e-5)
    assert not torch.allclose(la[:, 6:], lb[:, 6:], atol=1e-5)


def test_weights_are_randomly_initialised_not_identical_between_seeds():
    torch.manual_seed(1)
    m1 = MiniLM(tiny_model_cfg(vocab_size=10))
    torch.manual_seed(2)
    m2 = MiniLM(tiny_model_cfg(vocab_size=10))
    assert not torch.equal(m1.tok_emb.weight, m2.tok_emb.weight)


def test_parameter_count_counts_tied_weights_once():
    cfg = tiny_model_cfg(vocab_size=10)
    model = MiniLM(cfg)
    unique = {id(p): p.numel() for p in model.parameters()}
    assert model.count_parameters() == sum(unique.values())

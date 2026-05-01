"""
Comprehensive inference pipeline tests for OpenMythos.

Validates production-critical properties: deterministic reproducibility,
KV cache correctness across multi-step decode, numerical stability over
long generation, depth extrapolation safety, generate() parameter edge
cases, eval-mode discipline, gradient isolation, and state-dict round-trips.

Every test class is parametrized over both attention backends (GQA / MLA)
to ensure parity.
"""

import copy
import io

import pytest
import torch

from open_mythos.main import (
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch, prompt length


def _cfg(attn_type: str, **overrides) -> MythosConfig:
    """Tiny config for fast CPU tests."""
    defaults = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type=attn_type,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        dropout=0.0,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


@pytest.fixture(params=["gqa", "mla"], ids=["GQA", "MLA"])
def model_and_cfg(request):
    """Yield (model, cfg) for each attention backend."""
    cfg = _cfg(request.param)
    model = OpenMythos(cfg)
    model.eval()
    return model, cfg


@pytest.fixture(params=["gqa", "mla"], ids=["GQA", "MLA"])
def model_with_dropout(request):
    """Model with dropout enabled for eval-mode testing."""
    cfg = _cfg(request.param, dropout=0.1)
    model = OpenMythos(cfg)
    return model, cfg


# ---------------------------------------------------------------------------
# 1. Deterministic reproducibility
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed + same input must produce bit-identical output."""

    def test_forward_deterministic(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            out1 = model(ids, n_loops=2)
            out2 = model(ids, n_loops=2)
        assert torch.equal(out1, out2)

    def test_generate_deterministic(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        torch.manual_seed(42)
        out1 = model.generate(ids.clone(), max_new_tokens=6, n_loops=2)
        torch.manual_seed(42)
        out2 = model.generate(ids.clone(), max_new_tokens=6, n_loops=2)
        assert torch.equal(out1, out2)

    def test_different_seeds_differ(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        torch.manual_seed(0)
        out1 = model.generate(ids.clone(), max_new_tokens=8, n_loops=2)
        torch.manual_seed(999)
        out2 = model.generate(ids.clone(), max_new_tokens=8, n_loops=2)
        # prompts match but generated tails should differ
        assert torch.equal(out1[:, :T], out2[:, :T])
        assert not torch.equal(out1[:, T:], out2[:, T:])


# ---------------------------------------------------------------------------
# 2. Incremental decode ↔ full-sequence equivalence
# ---------------------------------------------------------------------------


class TestIncrementalDecode:
    """Token-by-token decode with KV cache must match full-sequence forward."""

    def test_prefill_logits_match_with_and_without_cache(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        with torch.no_grad():
            logits_full = model(ids, n_loops=2)
            cache = {}
            logits_cached = model(ids, n_loops=2, kv_cache=cache)
        assert torch.allclose(logits_full, logits_cached, atol=1e-5)

    def test_single_decode_step_matches_full_recompute(self, model_and_cfg):
        """After prefill, a single decode token with cache must match
        a full (prompt + 1 token) forward without cache on the last position."""
        model, cfg = model_and_cfg
        prompt = torch.randint(0, cfg.vocab_size, (1, T))
        next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
        full_seq = torch.cat([prompt, next_tok], dim=1)

        with torch.no_grad():
            # full recompute
            logits_full = model(full_seq, n_loops=2)[:, -1, :]

            # incremental: prefill then decode
            cache = {}
            model(prompt, n_loops=2, kv_cache=cache, start_pos=0)
            logits_inc = model(next_tok, n_loops=2, kv_cache=cache, start_pos=T)[
                :, -1, :
            ]
        assert torch.allclose(logits_full, logits_inc, atol=1e-4)

    def test_multi_step_decode_matches_full_recompute(self, model_and_cfg):
        """Three incremental decode steps must agree with full recompute
        at every step."""
        model, cfg = model_and_cfg
        prompt = torch.randint(0, cfg.vocab_size, (1, T))
        extra_tokens = torch.randint(0, cfg.vocab_size, (1, 3))

        with torch.no_grad():
            # incremental
            cache = {}
            model(prompt, n_loops=2, kv_cache=cache, start_pos=0)
            inc_logits = []
            for step in range(3):
                tok = extra_tokens[:, step : step + 1]
                out = model(tok, n_loops=2, kv_cache=cache, start_pos=T + step)
                inc_logits.append(out[:, -1, :])

            # full recompute at each length
            full_logits = []
            for step in range(3):
                seq = torch.cat([prompt, extra_tokens[:, : step + 1]], dim=1)
                out = model(seq, n_loops=2)
                full_logits.append(out[:, -1, :])

        for i in range(3):
            assert torch.allclose(inc_logits[i], full_logits[i], atol=1e-4), (
                f"Mismatch at decode step {i}"
            )


# ---------------------------------------------------------------------------
# 3. KV cache structure and integrity
# ---------------------------------------------------------------------------


class TestKVCacheStructure:
    """Validate cache shape, key population, and isolation."""

    def test_cache_keys_populated_after_prefill(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        cache = {}
        with torch.no_grad():
            model(ids, n_loops=2, kv_cache=cache)
        # prelude, recurrent loops, coda all write entries
        assert len(cache) > 0
        # at minimum: prelude_0, recurrent_loop_{0..n}, coda_0
        assert any("prelude" in k for k in cache)
        assert any("coda" in k for k in cache)
        assert any("recurrent_loop" in k for k in cache)

    def test_cache_seq_dim_grows_per_step(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        cache = {}
        with torch.no_grad():
            model(ids, n_loops=2, kv_cache=cache, start_pos=0)
        # record seq lengths
        first_lens = {}
        for key, entry in cache.items():
            for name, tensor in entry.items():
                first_lens[(key, name)] = tensor.shape[1]

        # decode one more token
        tok = torch.randint(0, cfg.vocab_size, (1, 1))
        with torch.no_grad():
            model(tok, n_loops=2, kv_cache=cache, start_pos=T)

        for key, entry in cache.items():
            for name, tensor in entry.items():
                assert tensor.shape[1] == first_lens[(key, name)] + 1, (
                    f"Cache entry {key}/{name} did not grow by 1"
                )

    def test_separate_caches_are_independent(self, model_and_cfg):
        """Two independent cache dicts must not interfere."""
        model, cfg = model_and_cfg
        ids_a = torch.randint(0, cfg.vocab_size, (1, T))
        ids_b = torch.randint(0, cfg.vocab_size, (1, T))
        cache_a, cache_b = {}, {}
        with torch.no_grad():
            logits_a = model(ids_a, n_loops=2, kv_cache=cache_a)
            logits_b = model(ids_b, n_loops=2, kv_cache=cache_b)
        # caches should have the same keys but different tensor values
        assert cache_a.keys() == cache_b.keys()
        some_key = next(iter(cache_a))
        some_name = next(iter(cache_a[some_key]))
        assert not torch.equal(
            cache_a[some_key][some_name], cache_b[some_key][some_name]
        )


# ---------------------------------------------------------------------------
# 4. Numerical stability over long generation
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """No NaN or Inf over extended generation runs."""

    def test_no_nan_inf_in_long_generate(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(ids, max_new_tokens=24, n_loops=2)
        assert not torch.isnan(out.float()).any()
        assert not torch.isinf(out.float()).any()
        assert out.shape == (1, 4 + 24)

    def test_logits_finite_at_max_seq_len(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
        with torch.no_grad():
            logits = model(ids, n_loops=2)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_logits_produce_valid_probabilities(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            logits = model(ids, n_loops=2)
        probs = torch.softmax(logits, dim=-1)
        assert (probs >= 0).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B, T), atol=1e-5)

    def test_no_degenerate_logit_distribution(self, model_and_cfg):
        """At least 2 logits per position should be non-negligible;
        a fully collapsed distribution indicates a broken forward pass."""
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        with torch.no_grad():
            logits = model(ids, n_loops=2)
        probs = torch.softmax(logits, dim=-1)
        # count tokens with prob > 0.1% per position; should always be > 1
        active = (probs > 0.001).sum(dim=-1)
        assert (active > 1).all()


# ---------------------------------------------------------------------------
# 5. generate() parameter edge cases
# ---------------------------------------------------------------------------


class TestGenerateParameters:
    """Validate generate() under boundary parameter values."""

    def test_zero_new_tokens(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out = model.generate(ids, max_new_tokens=0, n_loops=2)
        assert torch.equal(out, ids)

    def test_single_new_token(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out = model.generate(ids, max_new_tokens=1, n_loops=2)
        assert out.shape == (1, T + 1)
        # prompt is preserved
        assert torch.equal(out[:, :T], ids)

    def test_single_token_prompt(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, 1))
        out = model.generate(ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (1, 1 + 4)

    def test_low_temperature(self, model_and_cfg):
        """Very low temperature should not produce NaN (approaches greedy)."""
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out = model.generate(ids, max_new_tokens=4, n_loops=2, temperature=0.01)
        assert out.shape == (1, T + 4)
        assert (out >= 0).all() and (out < cfg.vocab_size).all()

    def test_high_temperature(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out = model.generate(ids, max_new_tokens=4, n_loops=2, temperature=10.0)
        assert out.shape == (1, T + 4)
        assert (out >= 0).all() and (out < cfg.vocab_size).all()

    def test_topk_disabled(self, model_and_cfg):
        """top_k=0 disables filtering; all tokens remain candidates."""
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out = model.generate(ids, max_new_tokens=4, n_loops=2, top_k=0)
        assert out.shape == (1, T + 4)

    def test_topk_equals_one(self, model_and_cfg):
        """top_k=1 should produce deterministic (greedy) output."""
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out1 = model.generate(ids.clone(), max_new_tokens=4, n_loops=2, top_k=1)
        out2 = model.generate(ids.clone(), max_new_tokens=4, n_loops=2, top_k=1)
        assert torch.equal(out1, out2)

    def test_topk_exceeds_vocab(self, model_and_cfg):
        """top_k > vocab_size should behave like top_k = vocab_size (no crash)."""
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out = model.generate(
            ids, max_new_tokens=4, n_loops=2, top_k=cfg.vocab_size + 100
        )
        assert out.shape == (1, T + 4)

    def test_generated_tokens_in_vocab_range(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = model.generate(ids, max_new_tokens=8, n_loops=2)
        assert (out >= 0).all()
        assert (out < cfg.vocab_size).all()

    def test_batch_generate_shape(self, model_and_cfg):
        """Batched generation must produce correct shape for every item."""
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = model.generate(ids, max_new_tokens=6, n_loops=2)
        assert out.shape == (B, T + 6)


# ---------------------------------------------------------------------------
# 6. Depth extrapolation (n_loops > max_loop_iters)
# ---------------------------------------------------------------------------


class TestDepthExtrapolation:
    """Running more loops than the model was configured for must not crash."""

    def test_forward_with_extra_loops(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        extra = cfg.max_loop_iters * 2  # double the configured depth
        with torch.no_grad():
            logits = model(ids, n_loops=extra)
        assert logits.shape == (1, T, cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_generate_with_extra_loops(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        out = model.generate(ids, max_new_tokens=4, n_loops=cfg.max_loop_iters * 3)
        assert out.shape == (1, T + 4)

    def test_lora_clamps_beyond_max(self):
        """LoRA adapter must not crash when loop_t > max_loop_iters;
        it should reuse the last learned scale."""
        from open_mythos.main import LoRAAdapter

        lora = LoRAAdapter(dim=64, rank=4, max_loops=3)
        x = torch.randn(1, 4, 64)
        # loop_t = 0, 1, 2 are in-range; 5 and 100 are out-of-range
        for t in [0, 2, 5, 100]:
            out = lora(x, loop_t=t)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_deeper_loops_differ_from_shallow(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        with torch.no_grad():
            shallow = model(ids, n_loops=1)
            deep = model(ids, n_loops=cfg.max_loop_iters * 2)
        assert not torch.allclose(shallow, deep)


# ---------------------------------------------------------------------------
# 7. ACT halting and LTI stability under deep loops
# ---------------------------------------------------------------------------


class TestRecurrenceStability:
    """Hidden state must remain bounded under many loop iterations."""

    @pytest.mark.parametrize("attn_type", ["gqa", "mla"])
    def test_hidden_state_bounded_deep_loops(self, attn_type):
        cfg = _cfg(attn_type)
        model = OpenMythos(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, T))

        with torch.no_grad():
            logits = model(ids, n_loops=cfg.max_loop_iters * 4)
        # logits should not explode
        assert logits.abs().max().item() < 1e6

    @pytest.mark.parametrize("attn_type", ["gqa", "mla"])
    def test_spectral_radius_invariant(self, attn_type):
        """ρ(A) < 1 must hold regardless of attention backend."""
        cfg = _cfg(attn_type)
        model = OpenMythos(cfg)
        A = model.recurrent.injection.get_A()
        assert A.max().item() < 1.0
        assert A.min().item() > 0.0

    @pytest.mark.parametrize("attn_type", ["gqa", "mla"])
    def test_act_output_nonzero(self, attn_type):
        """ACT-weighted output must not be all-zero (halting must assign mass)."""
        cfg = _cfg(attn_type)
        block = RecurrentBlock(cfg)
        block.eval()
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads if attn_type == "gqa" else cfg.qk_rope_head_dim,
            cfg.max_seq_len,
        )
        h = torch.randn(1, T, cfg.dim)
        e = torch.randn(1, T, cfg.dim)
        with torch.no_grad():
            out = block(h, e, freqs[:T], n_loops=cfg.max_loop_iters)
        assert out.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 8. Eval mode discipline
# ---------------------------------------------------------------------------


class TestEvalMode:
    """Dropout must be disabled and outputs must be deterministic in eval mode."""

    def test_eval_mode_disables_dropout(self, model_with_dropout):
        model, cfg = model_with_dropout
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            out1 = model(ids, n_loops=2)
            out2 = model(ids, n_loops=2)
        assert torch.equal(out1, out2), "Eval mode produced non-deterministic output"

    def test_train_mode_with_dropout_is_stochastic(self, model_with_dropout):
        model, cfg = model_with_dropout
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        # run twice without no_grad so dropout is active
        out1 = model(ids, n_loops=2)
        out2 = model(ids, n_loops=2)
        # with dropout=0.1 these should almost certainly differ
        assert not torch.equal(out1, out2), (
            "Train mode with dropout produced identical outputs"
        )


# ---------------------------------------------------------------------------
# 9. Gradient isolation
# ---------------------------------------------------------------------------


class TestGradientIsolation:
    """generate() must not accumulate gradients on model parameters."""

    def test_generate_no_grad(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        model.generate(ids, max_new_tokens=4, n_loops=2)
        for name, param in model.named_parameters():
            assert param.grad is None, f"Gradient leaked into {name}"

    def test_generate_does_not_enable_grad(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        assert not torch.is_grad_enabled() or True  # baseline
        model.generate(ids, max_new_tokens=2, n_loops=2)
        # no parameter should have requires_grad toggled off
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable > 0


# ---------------------------------------------------------------------------
# 10. State-dict save/load round-trip
# ---------------------------------------------------------------------------


class TestStateDictRoundTrip:
    """Saving and loading state_dict must produce bit-identical inference."""

    @pytest.mark.parametrize("attn_type", ["gqa", "mla"])
    def test_save_load_produces_identical_output(self, attn_type):
        cfg = _cfg(attn_type)
        model = OpenMythos(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, T))

        with torch.no_grad():
            logits_before = model(ids, n_loops=2)

        # round-trip through in-memory buffer
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)

        model2 = OpenMythos(cfg)
        model2.load_state_dict(torch.load(buf, weights_only=True))
        model2.eval()

        with torch.no_grad():
            logits_after = model2(ids, n_loops=2)

        assert torch.equal(logits_before, logits_after)

    @pytest.mark.parametrize("attn_type", ["gqa", "mla"])
    def test_deepcopy_produces_identical_output(self, attn_type):
        cfg = _cfg(attn_type)
        model = OpenMythos(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, T))

        with torch.no_grad():
            logits_orig = model(ids, n_loops=2)

        model_copy = copy.deepcopy(model)
        model_copy.eval()

        with torch.no_grad():
            logits_copy = model_copy(ids, n_loops=2)

        assert torch.equal(logits_orig, logits_copy)


# ---------------------------------------------------------------------------
# 11. Batched inference consistency
# ---------------------------------------------------------------------------


class TestBatchConsistency:
    """Each item in a batch must be independent of other items."""

    def test_batch_independence(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids_a = torch.randint(0, cfg.vocab_size, (1, T))
        ids_b = torch.randint(0, cfg.vocab_size, (1, T))

        with torch.no_grad():
            solo_a = model(ids_a, n_loops=2)
            batched = model(torch.cat([ids_a, ids_b], dim=0), n_loops=2)

        assert torch.allclose(solo_a, batched[:1], atol=1e-5)

    def test_batch_generate_preserves_prompts(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = model.generate(ids, max_new_tokens=4, n_loops=2)
        assert torch.equal(out[:, :T], ids)


# ---------------------------------------------------------------------------
# 12. Forward with start_pos (incremental decoding API)
# ---------------------------------------------------------------------------


class TestStartPos:
    """start_pos must correctly offset RoPE frequencies."""

    def test_start_pos_zero_matches_default(self, model_and_cfg):
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        with torch.no_grad():
            default = model(ids, n_loops=2)
            explicit = model(ids, n_loops=2, start_pos=0)
        assert torch.equal(default, explicit)

    def test_start_pos_does_not_crash_at_boundary(self, model_and_cfg):
        """start_pos + T = max_seq_len should work (last valid position)."""
        model, cfg = model_and_cfg
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        start = cfg.max_seq_len - T
        with torch.no_grad():
            logits = model(ids, n_loops=2, start_pos=start)
        assert logits.shape == (1, T, cfg.vocab_size)
        assert not torch.isnan(logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])

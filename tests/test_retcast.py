"""Unit tests for RetCast: RetNet2.0 modes, JEPA losses, RevIN, data splits, and TimesFM decoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from retcast import (
    MultiScaleRetention,
    RetCastConfig,
    RetCastJEPA,
    RevIN,
    retnet_state_bytes,
)
from retcast.baselines import (
    last_value_forecast,
    moving_average_forecast,
    seasonal_naive_forecast,
)
from retcast.data import ETTDataset, generate_synthetic_batch
from retcast.pretrain_data import (
    NON_ETT_CHRONOS_SUBSETS,
    NonETTPretrainCorpus,
    SOLAR_AL_URL,
    download_non_ett_corpus,
)
from retcast.metrics import (
    compute_all_metrics,
    effective_rank,
    latent_cosine_alignment,
    latent_retrieval_r1,
)
from retcast.timesfm_wrapper import PatchedTimeSeriesDecoder, TimesFMConfig

DT = torch.float64


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


MEMORIES = [
    ("none", 1.0, 0),
    ("delta", 1.0, 0),
    ("delta", 2.0, 0),
    ("none", 1.0, 4),
    ("delta", 2.0, 4),
]


def _close(a, b, atol=1e-7, rtol=1e-6):
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


# ------------------------------------------------------------------ RetNet2.0
@pytest.mark.parametrize("memory,beta_max,short_conv", MEMORIES)
@pytest.mark.parametrize("chunk", [3, 7, 16])
def test_retention_modes_match(chunk, memory, beta_max, short_conv):
    layer = (
        MultiScaleRetention(
            32, 4, memory=memory, beta_max=beta_max, short_conv=short_conv
        )
        .to(DT)
        .eval()
    )
    x = torch.randn(2, 23, 32, dtype=DT)
    par, par_state = layer(x, mode="parallel", return_state=True)
    rec, rec_state = layer(x, mode="recurrent")
    chk, chk_state = layer(x, mode="chunkwise", chunk_size=chunk)
    _close(par, rec)
    _close(par, chk)
    _close(par_state, rec_state)
    _close(par_state, chk_state)


@pytest.mark.parametrize("memory,beta_max,short_conv", MEMORIES)
@pytest.mark.parametrize("split", [1, 7, 15])
@pytest.mark.parametrize("mode2", ["parallel", "recurrent", "chunkwise"])
def test_retention_streaming(split, mode2, memory, beta_max, short_conv):
    layer = (
        MultiScaleRetention(
            32, 4, memory=memory, beta_max=beta_max, short_conv=short_conv
        )
        .to(DT)
        .eval()
    )
    x = torch.randn(2, 23, 32, dtype=DT)
    full, _ = layer(x, mode="parallel")
    a, state = layer(x[:, :split], mode="parallel", return_state=True)
    b, _ = layer(x[:, split:], state=state, offset=split, mode=mode2, chunk_size=4)
    _close(full, torch.cat([a, b], dim=1))


def test_delta_memory_overwrites():
    layer = MultiScaleRetention(32, 4, memory="delta", beta_max=1.0).to(DT).eval()
    H, dk, dv = 4, layer.head_k, layer.head_v
    k = F.normalize(torch.randn(1, H, 1, dk, dtype=DT), dim=-1).expand(1, H, 2, dk)
    v = torch.randn(1, H, 2, dv, dtype=DT)
    beta = torch.ones(1, H, 2, dtype=DT)
    _, S_delta = layer._recurrent(k, k, v, None, beta=beta)
    read_delta = (k[:, :, :1] @ S_delta).squeeze(2)
    _close(read_delta, v[:, :, 1], atol=1e-9, rtol=1e-9)

    _, S_plain = layer._recurrent(k, k, v, None, beta=None)
    read_plain = (k[:, :, :1] @ S_plain).squeeze(2)
    assert not torch.allclose(read_plain, read_delta)


# --------------------------------------------------------------- RetCast-JEPA
def test_preset_param_counts():
    bounds = {
        "2m": (1_500_000, 2_500_000),
        "5m": (4_000_000, 6_000_000),
        "9m": (8_000_000, 10_000_000),
    }
    for name, (lo, hi) in bounds.items():
        m = RetCastJEPA(RetCastConfig.from_preset(name))
        n = m.num_params(include_target_encoder=False)
        assert lo <= n <= hi, f"{name} online params {n:,} outside [{lo:,}, {hi:,}]"

        m_ret = RetCastJEPA(
            RetCastConfig.from_preset(name, arch="retnet", memory="none", short_conv=0)
        )
        m_tr = RetCastJEPA(
            RetCastConfig.from_preset(name, arch="transformer", short_conv=0)
        )
        assert m_ret.num_params() == m_tr.num_params()


def test_revin_invertibility_and_masking():
    revin = RevIN(eps=1e-6)
    x = torch.randn(3, 64) * 5.0 + 12.0
    pad = torch.zeros(3, 64)
    pad[:, :16] = 1.0
    # Inject extreme values in padded slots to ensure they are strictly ignored
    x_corrupt = x.clone()
    x_corrupt[:, :16] = 1e6

    x_norm, stats = revin.normalize(x_corrupt, padding=pad)
    assert torch.allclose(x_norm[:, :16], torch.zeros(3, 16))
    rec = revin.denormalize(x_norm, stats)
    torch.testing.assert_close(rec[:, 16:], x[:, 16:], atol=1e-4, rtol=1e-4)


def test_jepa_losses_and_ema_update():
    cfg = RetCastConfig.from_preset("tiny")
    model = RetCastJEPA(cfg)
    ctx = torch.randn(4, 128)
    fut = torch.randn(4, 96)
    freq = torch.zeros(4, dtype=torch.long)

    loss, metrics = model.compute_loss(ctx, fut, freq=freq)
    assert torch.isfinite(loss)
    for k in ("forecast_loss", "jepa_loss", "vicreg_loss"):
        assert np.isfinite(metrics[k]) and metrics[k] >= 0.0

    loss.backward()
    enc_grad = sum(
        p.grad.abs().sum().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    )
    pred_grad = sum(
        p.grad.abs().sum().item()
        for p in model.predictor.parameters()
        if p.grad is not None
    )
    assert enc_grad > 0.0
    assert pred_grad > 0.0
    assert all(p.grad is None for p in model.target_encoder.parameters())

    # Target weights must move toward online weights under EMA update
    before = [p.clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.5)
    model.update_target_encoder(tau=0.9)
    diffs = [
        (p_after - p_before).abs().max().item()
        for p_after, p_before in zip(model.target_encoder.parameters(), before)
    ]
    assert max(diffs) > 1e-3


def test_recurrent_latent_rollout_constant_state():
    cfg = RetCastConfig.from_preset("tiny")
    model = RetCastJEPA(cfg).eval()
    f_emb = model.freq_emb(torch.zeros(2, dtype=torch.long)).unsqueeze(1)

    with torch.no_grad():
        ctx_short = torch.randn(2, 64)
        p_short, pad_short = model._patchify(ctx_short)
        z_short = model.encoder(p_short, pad_short)
        h_short, st_short = model.forward_backbone(
            z_short + f_emb, mode="chunkwise", chunk_size=4, return_state=True
        )
        bytes_short = retnet_state_bytes(st_short)

        ctx_long = torch.randn(2, 512)
        p_long, pad_long = model._patchify(ctx_long)
        z_long = model.encoder(p_long, pad_long)
        h_long, st_long = model.forward_backbone(
            z_long + f_emb, mode="chunkwise", chunk_size=4, return_state=True
        )
        bytes_long = retnet_state_bytes(st_long)
        assert bytes_short == bytes_long

        z_roll_4, st_4 = model.rollout_latent_recurrent(
            h_long[:, -1:], st_long, offset=z_long.shape[1], n_patches=4, freq_emb=f_emb
        )
        z_roll_12, st_12 = model.rollout_latent_recurrent(
            h_long[:, -1:], st_long, offset=z_long.shape[1], n_patches=12, freq_emb=f_emb
        )
        assert z_roll_4.shape == (2, 4, cfg.d_model)
        assert z_roll_12.shape == (2, 12, cfg.d_model)
        assert retnet_state_bytes(st_4) == bytes_long
        assert retnet_state_bytes(st_12) == bytes_long


def test_causal_horizon_consistency():
    cfg = RetCastConfig.from_preset("tiny")
    model = RetCastJEPA(cfg).eval()
    ctx = torch.randn(2, 256)
    with torch.no_grad():
        out_96 = model(ctx, horizon_len=96)
        out_336 = model(ctx, horizon_len=336)
    torch.testing.assert_close(
        out_96["point"], out_336["point"][:, :96], atol=1e-5, rtol=1e-5
    )


def test_overfits_single_timeseries_batch():
    torch.manual_seed(7)
    cfg = RetCastConfig.from_preset("tiny", jepa_weight=0.25, vicreg_weight=0.05)
    model = RetCastJEPA(cfg)
    ctx_np, fut_np, freq_np = generate_synthetic_batch(
        8, context_len=256, horizon_len=96, rng=np.random.default_rng(7)
    )
    ctx = torch.from_numpy(ctx_np)
    fut = torch.from_numpy(fut_np)
    freq = torch.from_numpy(freq_np)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first_loss = None
    final_loss = None
    for _ in range(100):
        loss, _ = model.compute_loss(ctx, fut, freq=freq)
        if first_loss is None:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target_encoder()
        final_loss = loss.item()

    assert final_loss < 0.2 * first_loss


# ------------------------------------------------------------- Data & Leakage
def test_ett_splits_and_scaler_no_leakage(tmp_path):
    # Create synthetic ETTh1-sized CSV (14400 rows, 7 cols) with a deliberate shift in test
    n_rows = 14400
    t = np.arange(n_rows, dtype=np.float32)
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    data = {}
    for idx, c in enumerate(cols):
        vals = np.sin(2 * np.pi * t / 24.0 + idx) * (idx + 1.0) + 10.0 * idx
        # Shift test region significantly; train-only scaler must ignore this
        vals[11520:] += 500.0
        data[c] = vals
    df = pd.DataFrame(data)
    df.insert(0, "date", pd.date_range("2016-07-01", periods=n_rows, freq="h"))
    csv_path = tmp_path / "ETTh1.csv"
    df.to_csv(csv_path, index=False)

    ds = ETTDataset("etth1", csv_path=csv_path, normalize=True)
    raw_train = df[cols].to_numpy(dtype=np.float32)[:8640]
    np.testing.assert_allclose(ds.scaler.mean_, raw_train.mean(axis=0), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ds.scaler.scale_, raw_train.std(axis=0), rtol=1e-4, atol=1e-4)

    past_test, actual_test, freq_id = ds.get_eval_windows(
        split="test", context_len=512, pred_len=96
    )
    assert freq_id == 0
    assert past_test.shape[1] == 512
    assert actual_test.shape[1] == 96
    # First test window actuals come strictly from >= 11520 (shifted region > +10 std)
    assert np.min(actual_test) > 20.0

    # Sanity baselines and metrics work end-to-end
    sn = seasonal_naive_forecast(past_test, pred_len=96, seasonality=24)
    lv = last_value_forecast(past_test, pred_len=96)
    ma = moving_average_forecast(past_test, pred_len=96, window=24)
    assert sn.shape == actual_test.shape == lv.shape == ma.shape
    m = compute_all_metrics(sn, actual_test)
    assert set(m.keys()) == {"mse", "mae", "smape", "wape", "nrmse"}

    # Representation metrics check
    z = torch.randn(16, 6, 32)
    assert effective_rank(z) > 1.0
    assert latent_cosine_alignment(z, z) == pytest.approx(1.0, abs=1e-5)
    assert latent_retrieval_r1(z, z) == pytest.approx(1.0, abs=1e-5)


def test_zero_shot_corpus_disjoint_from_ett(tmp_path, monkeypatch):
    # Verify static source manifests exclude ETT completely
    for subset in NON_ETT_CHRONOS_SUBSETS:
        assert "ett" not in subset.lower()
    assert "ett" not in SOLAR_AL_URL.lower()

    with pytest.raises(ValueError, match="Refusing ETT"):
        download_non_ett_corpus(tmp_path, subsets=("etth1/train.parquet",), include_solar=False)

    # Create a local synthetic non-ETT parquet file with shifted trailing 20%
    t = np.arange(1500, dtype=np.float32)
    series_a = np.sin(2.0 * np.pi * t / 24.0)
    series_b = np.cos(2.0 * np.pi * t / 48.0)
    series_a[1200:] += 50.0  # shift in held-out 20% val region
    series_b[1200:] += 50.0

    pq_path = tmp_path / "mock_domain.parquet"
    pd.DataFrame({"id": ["s1", "s2"], "target": [series_a, series_b]}).to_parquet(
        pq_path
    )

    monkeypatch.setattr(
        "retcast.pretrain_data.download_non_ett_corpus",
        lambda **kw: {"mock_domain": pq_path},
    )
    corpus = NonETTPretrainCorpus(
        cache_dir=tmp_path,
        include_solar=False,
        context_len=512,
        horizon_len=96,
        synthetic_ratio=0.0,
        seed=0,
    )
    assert corpus.source_names == ["mock_domain"]
    assert all("ett" not in s.lower() for s in corpus.source_names)

    # Train sampling must stay inside the first 80% (unshifted ~[-1, 1])
    ctx, fut, _ = corpus.sample_batch(8, horizon_len=96, augment=False)
    assert ctx.shape == (8, 512) and fut.shape == (8, 96)
    assert float(fut.abs().max()) < 5.0

    # Validation windows come strictly from the trailing 20% (shifted > +10 std, clipped to 12.0)
    val_p, val_a, _ = corpus.get_val_windows(pred_len=96, max_windows=16)
    assert val_p.shape[1] == 512 and val_a.shape[1] == 96
    assert float(val_a.min()) > 10.0


def test_legacy_decoder_compatibility():
    cfg = RetCastConfig.from_preset("tiny", decoder_mode="legacy")
    m = RetCastJEPA(cfg).eval()
    out = m(torch.randn(2, 128), horizon_len=96)
    assert out["point"].shape == (2, 96)


def test_from_pretrained_safetensors_roundtrip(tmp_path):
    import json
    from safetensors.torch import save_file

    cfg = RetCastConfig.from_preset("tiny")
    m = RetCastJEPA(cfg).eval()
    x = torch.randn(2, 128)
    with torch.no_grad():
        ref = m(x, horizon_len=96)["point"]

    # Export in SafeTensors directory layout under alias target name
    v_dir = tmp_path / "zs_retcast_2m_jepa"
    v_dir.mkdir(parents=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()},
        str(v_dir / "model.safetensors"),
    )
    (v_dir / "config.json").write_text(json.dumps({"model": cfg.to_dict()}))

    loaded = RetCastJEPA.from_pretrained(tmp_path, variant="2m", device="cpu")
    with torch.no_grad():
        got = loaded(x, horizon_len=96)["point"]
    assert torch.allclose(ref, got, atol=1e-6)

    # Also test direct .pt checkpoint loading
    pt_path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg.to_dict()}, pt_path)
    loaded_pt = RetCastJEPA.from_pretrained(pt_path, device="cpu")
    with torch.no_grad():
        got_pt = loaded_pt(x, horizon_len=96)["point"]
    assert torch.allclose(ref, got_pt, atol=1e-6)


# -------------------------------------------------------------------- TimesFM
def test_timesfm_decoder_shapes():
    cfg = TimesFMConfig(
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        patch_len=32,
        horizon_len=128,
    )
    decoder = PatchedTimeSeriesDecoder(cfg).eval()
    B, C, H = 2, 128, 192
    x = torch.randn(B, C)
    pad = torch.zeros(B, C + H)
    freq = torch.zeros(B, 1, dtype=torch.long)
    with torch.no_grad():
        point, full = decoder.decode(x, pad, freq, horizon_len=H)
    assert point.shape == (B, H)
    assert full.shape == (B, H, 1 + len(cfg.quantiles))

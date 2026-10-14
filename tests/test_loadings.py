import numpy as np
import pandas as pd
import pytest

from crisispaths.loadings import Loadings, estimate_loadings

FACTORS = ["EQ", "TECH", "USD"]
UNAFFECTED_TEXT = (
    "this asset is modelled as unaffected by the scenario, which is a model claim, not a fact"
)


def factor_frame(n, seed=0, start="2010-01-04"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "EQ": rng.normal(0, 0.01, n),
            "TECH": rng.normal(0, 0.008, n),
            "USD": rng.normal(0, 0.004, n),
        },
        index=idx,
    )


def panel(n, betas, noise=0.01, seed=1):
    f = factor_frame(n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    out = {}
    for a, b in betas.items():
        out[a] = f[FACTORS].to_numpy() @ np.array([b.get(k, 0.0) for k in FACTORS]) + rng.normal(
            0, noise, n
        )
    return pd.DataFrame(out, index=f.index), f


def windows_for(idx, frac=0.5):
    cut = idx[int(len(idx) * frac)]
    return {
        "calm": [(idx[0], idx[int(len(idx) * frac) - 1])],
        "stress": [(cut, idx[-1])],
    }


TRUE = {"A": {"EQ": 1.2, "TECH": 0.3, "USD": -0.5}, "B": {"EQ": 0.6, "USD": 1.0}}


@pytest.mark.parametrize("n", [120, 500, 5000])  # 60 / 250 / 2500 observations per regime
def test_recovery_within_2se(n):
    """Estimates sit within 2 standard errors of the truth at about the nominal rate."""
    hits = []
    for seed in range(25):
        rets, f = panel(n, TRUE, seed=seed)
        L = estimate_loadings(rets, f, windows_for(rets.index), dimson=False)
        assert isinstance(L, Loadings)
        for regime in ("calm", "stress"):
            for a, b in TRUE.items():
                for k in FACTORS:
                    est = L.betas[regime].loc[a, k]
                    se = L.std_errors[regime].loc[a, k]
                    hits.append(abs(est - b.get(k, 0.0)) < 2.0 * se)
    assert np.mean(hits) >= 0.90


def test_se_shrinks_and_shrinkage_decreases_with_n():
    small, fs = panel(240, TRUE, seed=5)
    big, fb = panel(5000, TRUE, seed=5)
    Ls = estimate_loadings(small, fs, windows_for(small.index), dimson=False)
    Lb = estimate_loadings(big, fb, windows_for(big.index), dimson=False)
    assert Lb.shrinkage.loc["A", "stress"] < Ls.shrinkage.loc["A", "stress"]
    assert Lb.std_errors["calm"].loc["A", "EQ"] < Ls.std_errors["calm"].loc["A", "EQ"]
    assert Lb.n_obs.loc["A", "calm"] > Ls.n_obs.loc["A", "calm"]


def test_shrinkage_bias_decreases_with_n():
    """Average signed bias toward the calm beta shrinks as the stress sample grows."""

    def mean_bias(n_stress):
        n_calm, out = 1000, []
        for seed in range(60):
            rng = np.random.default_rng(seed)
            f = factor_frame(n_calm + n_stress, seed=1000 + seed)
            beta = np.r_[np.full(n_calm, 1.0), np.full(n_stress, 2.0)]
            y = beta * f["EQ"].to_numpy() + rng.normal(0, 0.01, len(f))
            rets = pd.DataFrame({"A": y}, index=f.index)
            w = {
                "calm": [(f.index[0], f.index[n_calm - 1])],
                "stress": [(f.index[n_calm], f.index[-1])],
            }
            L = estimate_loadings(rets, f[["EQ"]], w, dimson=False, k=20.0)
            out.append(2.0 - L.betas["stress"].loc["A", "EQ"])
        return float(np.mean(out))

    small, large = mean_bias(30), mean_bias(500)
    assert small > 0  # shrunk toward the calm beta
    assert large < small


def test_zero_exposure_flagged_and_zeroed():
    rets, f = panel(500, TRUE, seed=11)
    rng = np.random.default_rng(12)
    rets["NOISE"] = rng.normal(0, 0.01, len(rets))
    L = estimate_loadings(rets, f, windows_for(rets.index))
    assert "NOISE" in L.unaffected
    assert "A" not in L.unaffected
    assert (L.betas["stress"].loc["NOISE"] == 0).all()
    assert (L.betas["calm"].loc["NOISE"] == 0).all()
    assert any(UNAFFECTED_TEXT in w and "NOISE" in w for w in L.warnings)


def test_override_recorded_and_applied():
    rets, f = panel(500, TRUE, seed=13)
    L = estimate_loadings(
        rets, f, windows_for(rets.index), overrides={"A": {"EQ": 2.0}}
    )
    assert L.betas["stress"].loc["A", "EQ"] == 2.0
    assert L.betas["calm"].loc["A", "EQ"] == 2.0
    assert L.metadata["overrides"] == {"A": {"EQ": 2.0}}


def test_override_unknown_asset_or_factor():
    rets, f = panel(300, TRUE)
    with pytest.raises(KeyError):
        estimate_loadings(rets, f, windows_for(rets.index), overrides={"Z": {"EQ": 1}})
    with pytest.raises(KeyError):
        estimate_loadings(rets, f, windows_for(rets.index), overrides={"A": {"ZZ": 1}})


def test_low_n_warning():
    rets, f = panel(80, TRUE)
    L = estimate_loadings(rets, f, windows_for(rets.index))
    assert any("n_obs" in w and "60" in w for w in L.warnings)
    rets2, f2 = panel(1000, TRUE)
    L2 = estimate_loadings(rets2, f2, windows_for(rets2.index))
    assert not any("n_obs" in w for w in L2.warnings)


def test_knn_fallback_for_asset_without_stress_data():
    n = 1000
    f = factor_frame(n, seed=21)
    rng = np.random.default_rng(22)
    # twins have calm beta 1.0, stress beta 2.0; far asset has calm 5.0, stress 0.5
    spec = {"T1": (1.0, 2.0), "T2": (1.05, 2.1), "T3": (0.95, 1.9), "FAR": (5.0, 0.5)}
    half = n // 2
    cols = {}
    for a, (c, s) in spec.items():
        beta = np.r_[np.full(half, c), np.full(n - half, s)]
        cols[a] = beta * f["EQ"].to_numpy() + rng.normal(0, 0.003, n)
    new = np.r_[np.full(half, 1.02), np.full(n - half, np.nan)] * f["EQ"].to_numpy()
    new[:half] += rng.normal(0, 0.003, half)
    cols["NEW"] = new
    rets = pd.DataFrame(cols, index=f.index)
    L = estimate_loadings(
        rets, f[["EQ"]], windows_for(rets.index), dimson=False, n_neighbors=3
    )
    assert "NEW" in L.imputed
    assert L.betas["stress"].loc["NEW", "EQ"] == pytest.approx(1.02 * 2.0, rel=0.1)
    assert any("NEW" in w and "nearest" in w for w in L.warnings)
    assert L.n_obs.loc["NEW", "stress"] == 0


def test_dimson_recovers_lagged_asset():
    n = 3000
    f = factor_frame(n, seed=31)
    rng = np.random.default_rng(32)
    lagged = 0.9 * f["EQ"].shift(1).fillna(0).to_numpy() + rng.normal(0, 0.003, n)
    rets = pd.DataFrame({"LAG": lagged}, index=f.index)
    w = windows_for(rets.index)
    plain = estimate_loadings(rets, f[["EQ"]], w, dimson=False)
    dim = estimate_loadings(rets, f[["EQ"]], w, dimson=True)
    assert abs(plain.betas["calm"].loc["LAG", "EQ"]) < 0.2
    assert dim.betas["calm"].loc["LAG", "EQ"] == pytest.approx(0.9, abs=0.1)
    assert "LAG" not in dim.unaffected


def test_determinism():
    rets, f = panel(400, TRUE, seed=41)
    w = windows_for(rets.index)
    a = estimate_loadings(rets, f, w)
    b = estimate_loadings(rets, f, w)
    pd.testing.assert_frame_equal(a.betas["stress"], b.betas["stress"])
    pd.testing.assert_frame_equal(a.std_errors["calm"], b.std_errors["calm"])
    assert a.warnings == b.warnings


def test_bad_inputs():
    rets, f = panel(200, TRUE)
    with pytest.raises(ValueError):
        estimate_loadings(rets, f, {"calm": [(rets.index[0], rets.index[-1])]})
    with pytest.raises(ValueError):
        estimate_loadings(rets.iloc[:, :0], f, windows_for(rets.index))


# --- imputation by one scalar stress amplification --------------------------------------------


def _impute_panel(neigh_calm, neigh_stress, new_calm, n=1600, seed=31):
    """Neighbours N1..N3 have a calm and a stress beta vector on (EQ, TECH); NEW has no stress."""
    f = factor_frame(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    half = n // 2
    cols = {}
    X = f[["EQ", "TECH"]].to_numpy()
    for j in range(3):
        c = np.array(neigh_calm) * (1 + 0.02 * j)
        s = np.array(neigh_stress) * (1 + 0.02 * j)
        b = np.vstack([np.tile(c, (half, 1)), np.tile(s, (n - half, 1))])
        cols[f"N{j + 1}"] = (X * b).sum(axis=1) + rng.normal(0, 0.003, n)
    far = np.vstack([np.tile([6.0, 6.0], (half, 1)), np.tile([6.0, 6.0], (n - half, 1))])
    cols["FAR"] = (X * far).sum(axis=1) + rng.normal(0, 0.003, n)
    new = X @ np.array(new_calm) + rng.normal(0, 0.003, n)
    new[half:] = np.nan
    cols["NEW"] = new
    return pd.DataFrame(cols, index=f.index), f[["EQ", "TECH"]]


def test_imputation_never_flips_a_calm_sign():
    # the neighbours' calm TECH beta is tiny and its stress beta large and negative: the old
    # per-factor ratio (-4) flipped NEW's TECH sign; one pooled scalar cannot
    rets, f = _impute_panel([1.0, 0.1], [1.2, -0.4], [1.0, 0.8])
    L = estimate_loadings(rets, f, windows_for(rets.index), dimson=False, n_neighbors=3)
    assert "NEW" in L.imputed
    calm = L.betas["calm"].loc["NEW"]
    stress = L.betas["stress"].loc["NEW"]
    assert calm["TECH"] > 0 and stress["TECH"] > 0 and stress["EQ"] > 0
    info = L.metadata["imputation"]["NEW"]
    assert 0.5 <= info["amplification"] <= 2.0
    assert set(info["neighbours"]) == {"N1", "N2", "N3"}
    np.testing.assert_allclose(stress.to_numpy(), info["amplification"] * calm.to_numpy())
    assert L.std_errors["stress"].loc["NEW"].isna().all()
    assert any("NEW" in w and "amplification" in w for w in L.warnings)


def test_amplification_is_clipped_both_ways():
    from crisispaths import loadings as ld

    rets, f = _impute_panel([1.0, 0.5], [4.0, 2.0], [1.0, 0.5])
    L = estimate_loadings(rets, f, windows_for(rets.index), dimson=False, n_neighbors=3)
    info = L.metadata["imputation"]["NEW"]
    assert info["amplification"] == ld.AMPLIFICATION_MAX and info["clipped"]
    rets, f = _impute_panel([1.0, 0.5], [0.1, 0.05], [1.0, 0.5])
    L = estimate_loadings(rets, f, windows_for(rets.index), dimson=False, n_neighbors=3)
    info = L.metadata["imputation"]["NEW"]
    assert info["amplification"] == ld.AMPLIFICATION_MIN and info["clipped"]
    calm_eq = L.betas["calm"].loc["NEW", "EQ"]
    assert L.betas["stress"].loc["NEW", "EQ"] == pytest.approx(0.5 * calm_eq)


def test_amplification_helper_fallback_and_formula():
    from crisispaths import loadings as ld

    s, clipped, degenerate = ld.stress_amplification(np.zeros((3, 2)), np.ones((3, 2)))
    assert (s, degenerate) == (1.0, True)
    calm = np.array([[1.0, 0.1], [1.0, 0.1]])
    stress = np.array([[1.5, -0.4], [1.5, -0.4]])
    s, clipped, degenerate = ld.stress_amplification(calm, stress)
    assert s == pytest.approx(float((stress * calm).sum() / (calm**2).sum()))
    assert not degenerate and not clipped
    assert ld.AMPLIFICATION_MIN == 0.5 and ld.AMPLIFICATION_MAX == 2.0

"""Score-level matching metrics (U11: paired bootstrap)."""

import numpy as np
import pytest


# -- paired bootstrap on the EER difference (U11) -------------------------------------

def _two_conditions(gap: float, n: int = 400, seed: int = 1):
    """Aligned scores where condition B separates better than A by `gap`."""
    rng = np.random.default_rng(seed)
    gen_a = rng.normal(60, 12, n)
    imp_a = rng.normal(40, 12, n * 4)
    # B shares A's pair noise (paired!), shifted toward separation by `gap`
    gen_b = gen_a + gap
    imp_b = imp_a - gap
    return gen_a, imp_a, gen_b, imp_b


def test_paired_difference_on_identical_conditions_is_exactly_zero():
    from fpe.metrics.matching import paired_bootstrap_eer_difference

    g, i, _, _ = _two_conditions(0.0)
    d = paired_bootstrap_eer_difference(g, i, g, i, n_resamples=200)
    assert d.difference == 0.0
    assert d.ci_low <= 0.0 <= d.ci_high and not d.excludes_zero


def test_paired_difference_resolves_a_large_constructed_gap():
    from fpe.metrics.matching import paired_bootstrap_eer_difference

    d = paired_bootstrap_eer_difference(*_two_conditions(8.0), n_resamples=400)
    assert d.difference < 0 and d.excludes_zero  # B better; interval excludes zero


def test_paired_interval_is_narrower_than_the_marginal_comparison():
    """The entire justification for pairing, asserted rather than assumed."""
    from fpe.metrics.matching import bootstrap_eer, paired_bootstrap_eer_difference

    ga, ia, gb, ib = _two_conditions(1.5)
    d = paired_bootstrap_eer_difference(ga, ia, gb, ib, n_resamples=400)
    a = bootstrap_eer(ga, ia, n_resamples=400)
    b = bootstrap_eer(gb, ib, n_resamples=400)
    marginal_width = (a.eer_ci_high - a.eer_ci_low) + (b.eer_ci_high - b.eer_ci_low)
    assert (d.ci_high - d.ci_low) < marginal_width


def test_resampling_is_paired_within_each_draw():
    """Add condition-independent pair noise: paired differencing must cancel it."""
    from fpe.metrics.matching import paired_bootstrap_eer_difference

    rng = np.random.default_rng(7)
    pair_noise_g = rng.normal(0, 25, 300)      # huge shared per-pair effect
    pair_noise_i = rng.normal(0, 25, 1200)
    gen = 60 + pair_noise_g
    imp = 40 + pair_noise_i
    d = paired_bootstrap_eer_difference(gen, imp, gen + 0.5, imp - 0.5, n_resamples=300)
    # the shared noise makes the marginal EERs awful and volatile, but the paired
    # interval of the difference stays tight around a small negative value
    assert d.ci_high - d.ci_low < 0.05
    assert d.difference <= 0.0


def test_mismatched_pair_sets_are_refused():
    from fpe.metrics.matching import paired_bootstrap_eer_difference

    g, i, gb, ib = _two_conditions(1.0)
    with pytest.raises(ValueError, match="identical pair sets"):
        paired_bootstrap_eer_difference(g[:-1], i, gb, ib)


def test_oversized_templates_are_trimmed_by_quality_for_bozorth3(tmp_path):
    """One hallucinated template must not kill a whole -M batch (U12, minex)."""
    from fpe.metrics.nbis import BOZORTH_MAX_MINUTIAE, _within_bozorth_limit

    big = tmp_path / "big.xyt"
    big.write_text("\n".join(f"{i} {i} 0 {i % 100}" for i in range(230)) + "\n",
                   encoding="ascii")
    trimmed = _within_bozorth_limit(big)
    assert trimmed != big and trimmed.is_file()
    rows = trimmed.read_text().splitlines()
    assert len(rows) == BOZORTH_MAX_MINUTIAE
    all_q = sorted((i % 100 for i in range(230)), reverse=True)
    assert min(int(r.split()[3]) for r in rows) == all_q[BOZORTH_MAX_MINUTIAE - 1]
    small = tmp_path / "small.xyt"
    small.write_text("1 2 3 4\n", encoding="ascii")
    assert _within_bozorth_limit(small) == small        # untouched below the limit

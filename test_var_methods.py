"""VaR without the normality assumption.

VaR and ES were computed from a closed-form normal distribution. Crypto
returns are not normal, and the error is not symmetric — it understates
exactly the tail the whole project is about, so the figures were a lower
bound on tail risk presented as a risk estimate.

Three estimators now run side by side: the normal one as a baseline to
measure against, the empirical quantile, and a Cornish-Fisher expansion that
corrects the normal quantile using the sample's own skew and excess kurtosis.
"""

import math
import random

import pytest

import var_analysis as V


def normal_sample(n=4000, sigma=0.02, seed=11):
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def fat_tailed_sample(seed=7):
    """Calm most days, occasionally violent — the shape that breaks normal VaR."""
    rng = random.Random(seed)
    return ([rng.gauss(0, 0.02) for _ in range(85)]
            + [-0.18, -0.22, -0.11, 0.14, -0.09])


def test_moments_of_a_normal_sample_are_near_zero():
    _mu, sigma, skew, kurt = V.sample_moments(normal_sample())
    assert sigma == pytest.approx(0.02, rel=0.1)
    assert abs(skew) < 0.15
    assert abs(kurt) < 0.3


def test_cornish_fisher_reduces_to_normal_without_skew_or_kurtosis():
    """The expansion must be a correction, not a distortion.

    With zero skew and zero excess kurtosis every correction term vanishes and
    CF should land on the closed-form normal answer. If it does not, the
    expansion is wrong rather than the data being interesting.
    """
    sample = normal_sample()
    n_var, _n_es, _ = V.compute_var_es(sample, 0.95)
    c_var, _c_es, skew, kurt = V.cornish_fisher_var_es(sample, 0.95)
    assert abs(skew) < 0.15 and abs(kurt) < 0.3
    assert c_var == pytest.approx(n_var, rel=0.05)


def test_fat_tails_make_the_normal_estimate_the_mildest():
    """The bug in one assertion.

    On a left-skewed, heavy-tailed sample the normal model must come out
    optimistic against an estimator that uses the actual third and fourth
    moments. If normal were the harshest, there would be nothing to fix.
    """
    sample = fat_tailed_sample()
    _mu, _sigma, skew, kurt = V.sample_moments(sample)
    assert skew < -1.0, "sample should be left-skewed"
    assert kurt > 3.0, "sample should be heavy-tailed"

    n_var, n_es, _ = V.compute_var_es(sample, 0.95)
    c_var, c_es, _, _ = V.cornish_fisher_var_es(sample, 0.95)

    assert c_var < n_var, "Cornish-Fisher VaR should be worse than normal"
    assert c_es < n_es, "Cornish-Fisher ES should be worse than normal"
    # and materially so, not by a rounding error
    assert (n_es - c_es) / abs(n_es) > 0.25


def test_expected_shortfall_is_never_milder_than_var():
    """ES averages the losses beyond VaR, so it cannot be the smaller loss.

    Holds for every estimator and every confidence level — a cheap invariant
    that catches a sign error or a mis-set integration range.
    """
    for sample in (normal_sample(), fat_tailed_sample()):
        for confidence in (0.90, 0.95, 0.99):
            m = V.compare_var_methods(sample, confidence)
            for method in ("normal", "historical", "cornish_fisher"):
                var_s, es_s = m[method]
                assert es_s <= var_s + 1e-9, (
                    f"{method} at {confidence}: ES {es_s} milder than VaR {var_s}")


def test_deeper_confidence_means_a_worse_loss():
    for sample in (normal_sample(), fat_tailed_sample()):
        m95 = V.compare_var_methods(sample, 0.95)
        m99 = V.compare_var_methods(sample, 0.99)
        for method in ("normal", "cornish_fisher"):
            assert m99[method][0] <= m95[method][0] + 1e-9


def test_historical_reports_how_thin_its_tail_is():
    """A 99% quantile from 90 observations rests on one day. Say so.

    The count is returned precisely so the caller can refuse to quote it.
    """
    sample = fat_tailed_sample()          # 90 observations
    _var, _es, tail_n = V.historical_var_es(sample, 0.99)
    assert tail_n <= 1

    _var, _es, tail_n_95 = V.historical_var_es(sample, 0.95)
    assert tail_n_95 >= 4


def test_historical_cannot_see_past_the_worst_observed_day():
    """Its defining limitation, pinned so nobody mistakes it for a strength."""
    sample = fat_tailed_sample()
    var_s, es_s, _ = V.historical_var_es(sample, 0.99)
    assert var_s >= min(sample) - 1e-12
    assert es_s >= min(sample) - 1e-12


def test_compare_returns_every_method_and_its_diagnostics():
    m = V.compare_var_methods(fat_tailed_sample(), 0.95)
    for key in ("normal", "historical", "cornish_fisher"):
        assert len(m[key]) == 2
    for key in ("sigma", "skew", "excess_kurtosis", "tail_observations",
                "sample"):
        assert key in m
    assert m["sample"] == 90


def test_empty_input_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        V.historical_var_es([], 0.95)

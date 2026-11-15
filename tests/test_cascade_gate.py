import random

import pytest

from network_orch.cascade.gate import (AdaptiveGate, calibrate, calibrate_worst_group,
                                       cp_upper)


def test_cp_upper_monotone_and_edges():
    assert cp_upper(0, 0) == 1.0
    assert cp_upper(0, 100) < cp_upper(5, 100) < 1.0
    assert cp_upper(10, 10) == 1.0


def _synthetic(n=2000, seed=0):
    rng = random.Random(seed)
    p, y = [], []
    for _ in range(n):
        unsafe = rng.random() < 0.3
        p.append(min(1, max(0, rng.gauss(0.85 if unsafe else 0.05, 0.1))))
        y.append(int(unsafe))
    return p, y


def test_calibrate_certifies_and_bound_holds():
    p, y = _synthetic()
    cert = calibrate(p, y, alpha=0.05)
    assert cert.certified and cert.upper_bound <= 0.05
    assert cert.allow_rate > 0.5


def test_calibrate_reports_uncertifiable():
    rng = random.Random(1)
    p = [rng.random() for _ in range(300)]
    y = [int(rng.random() < 0.5) for _ in range(300)]  # uninformative scores
    assert not calibrate(p, y, alpha=0.01).certified


def test_adaptive_gate_tightens_on_miss_and_requires_certificate():
    p, y = _synthetic()
    g = AdaptiveGate(calibrate(p, y, alpha=0.1))
    t0 = g.tau
    g.observe(was_unsafe=True)
    assert g.tau < t0
    with pytest.raises(ValueError):
        AdaptiveGate(calibrate([0.4] * 50, [1] * 50, alpha=0.01))


def test_worst_group_is_at_least_as_conservative_as_pooled():
    rng = random.Random(3)
    p, y, g = [], [], []
    for i in range(1500):
        grp = "a" if i % 2 else "b"
        unsafe = rng.random() < 0.3
        # group b has weaker separation: unsafe items often score low
        mu = 0.85 if unsafe else 0.05
        if grp == "b" and unsafe and rng.random() < 0.5:
            mu = 0.05
        p.append(min(1, max(0, rng.gauss(mu, 0.08))))
        y.append(int(unsafe)); g.append(grp)
    pooled = calibrate(p, y, alpha=0.12)
    robust = calibrate_worst_group(p, y, g, alpha=0.12)
    assert pooled.certified
    assert (not robust.certified) or robust.tau <= pooled.tau

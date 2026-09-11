"""Property-based tests (Hypothesis) for finsim's tax math and engine.

Where the brute-force tests check known points, these check LAWS over
the whole input space: round-trips, monotonicity, bounds, and exactness
of the piecewise machinery for arbitrary incomes, SS levels, deflators,
and penalties. Hypothesis shrinks any counterexample to a minimal case.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from finsim import tax

money = st.floats(min_value=0, max_value=2_000_000,
                  allow_nan=False, allow_infinity=False)
small_money = st.floats(min_value=0, max_value=600_000,
                        allow_nan=False, allow_infinity=False)


@given(net=st.floats(1, 500_000), other=small_money,
       pen=st.sampled_from([0.0, 0.10]))
@settings(max_examples=200, deadline=None)
def test_gross_up_round_trips(net, other, pen):
    gross = tax.gross_up(net, other_income=other, penalty=pen)
    tax_due = (tax.total_ordinary(other + gross)
               - tax.total_ordinary(other))
    assert gross - tax_due - pen * gross == pytest.approx(net, abs=1.0)
    assert gross >= net - 1e-9


@given(x=money, y=money)
@settings(max_examples=200, deadline=None)
def test_ordinary_tax_monotone_and_bounded(x, y):
    lo, hi = sorted((x, y))
    t_lo, t_hi = tax.total_ordinary(lo), tax.total_ordinary(hi)
    assert t_hi >= t_lo - 1e-9
    # marginal never exceeds top fed + MA + surtax
    assert t_hi - t_lo <= (hi - lo) * (0.37 + 0.09) + 1e-6
    assert t_lo >= 0


@given(w=small_money,
       ss=st.floats(0, 90_000),
       deflator=st.floats(0.2, 1.0))
@settings(max_examples=300, deadline=None)
def test_ss_grid_exact_everywhere(w, ss, deflator):
    """The piecewise grid equals direct evaluation for ANY point —
    this is the property that caught the phase-in cap-kink bug."""
    x, T = tax.ordinary_tax_grid(ss_total=ss, ss_deflator=deflator)
    fed_taxable = (w + tax.taxable_ss_of(w, ss, deflator=deflator)
                   - tax.STANDARD_DEDUCTION["mfj"])
    brute = tax.tax_from_brackets(fed_taxable, tax.FEDERAL["mfj"]) + tax.ma_tax(w)
    assert float(np.interp(w, x, T)) == pytest.approx(brute, abs=1.0)


@given(other=small_money, ss=st.floats(0, 90_000))
@settings(max_examples=200, deadline=None)
def test_ss_taxable_fraction_bounds_and_monotone(other, ss):
    f = tax.ss_taxable_fraction(other, ss)
    assert 0.0 <= f <= 0.85 + 1e-12
    assert tax.ss_taxable_fraction(other + 10_000, ss) >= f - 1e-12


@given(gain=st.floats(0, 1_000_000), ordinary=small_money)
@settings(max_examples=200, deadline=None)
def test_ltcg_bounds_and_stacking_monotone(gain, ordinary):
    t = tax.ltcg_tax(gain, ordinary)
    assert 0.0 <= t <= gain * 0.20 + 1e-6
    # more ordinary income can only push gains into higher bands
    assert tax.ltcg_tax(gain, ordinary + 50_000) >= t - 1e-6


@given(estate=st.floats(0, 30_000_000))
@settings(max_examples=200, deadline=None)
def test_ma_estate_tax_laws(estate):
    t = tax.ma_estate_tax(estate)
    assert t >= 0
    assert t == 0 or estate > tax.MA_ESTATE_EXCLUSION
    # monotone with bounded marginal
    t2 = tax.ma_estate_tax(estate + 100_000)
    assert 0 <= t2 - t <= 100_000 * 0.16 + 1e-6


@given(magi=st.floats(0, 400_000), premium=st.floats(1_000, 40_000))
@settings(max_examples=200, deadline=None)
def test_aca_cost_bounded_by_premium(magi, premium):
    c = tax.aca_premium_cost(magi, premium)
    assert 0 <= c <= premium + 1e-9
    # above the cliff it's exactly the full premium
    if magi > 4.0 * tax.FPL_2:
        assert c == premium


@given(net=st.floats(1, 300_000), other=small_money,
       ss=st.floats(0, 60_000))
@settings(max_examples=150, deadline=None)
def test_grid_inversion_round_trips(net, other, ss):
    """The engine's H-function inversion (gross from net) must round-
    trip against the same grid, at any starting income level."""
    x, T = tax.ordinary_tax_grid(ss_total=ss)
    pen = 0.10
    Hx = x * (1 - pen) - T
    H0 = float(np.interp(other, x, Hx))
    total = float(np.interp(H0 + net, Hx, x))
    gross = total - other
    got_net = (gross * (1 - pen)
               - (float(np.interp(other + gross, x, T))
                  - float(np.interp(other, x, T))))
    assert got_net == pytest.approx(net, abs=1.0)
    assert gross >= 0


# ── engine-level properties ─────────────────────────────────────────────

@given(taxable=st.floats(0, 3_000_000), pretax=st.floats(0, 3_000_000),
       spend=st.floats(0, 300_000))
@settings(max_examples=40, deadline=None)
def test_engine_accounting_identities(taxable, pretax, spend):
    """For arbitrary snapshots and spending: no NaNs, penalties only
    when pre-tax-ish money moved pre-60, estate no greater than what a
    tax-free liquidation would give."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_engine import buckets, det, mini_cfg

    cfg = mini_cfg(retirement={"retirement_year": 2026},
                   spending={"retirement_base": spend},
                   strategy={"name": "taxable_first"},
                   taxes={"mode": "brackets"})
    res = det(cfg, buckets(taxable=taxable, pretax=pretax))
    assert np.isfinite(res["estate"]).all()
    assert res["wd_tax_paid"][0] >= -1e-6
    assert res["penalty_paid"][0] >= -1e-6
    gross_wealth = taxable + pretax
    assert res["estate"][0] <= gross_wealth + 1e-6
    # penalties imply pre-tax draws happened
    if res["penalty_paid"][0] > 1:
        assert sum(t["draw_pretax"] + t["draw_roth"]
                   for t in res["trace"]) > 0

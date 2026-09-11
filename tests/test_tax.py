"""Tests for finsim.tax — bracket math, LTCG stacking, MA, gross-up."""

import pytest

from finsim import tax


def test_zero_and_negative_income():
    assert tax.tax_from_brackets(0, tax.FEDERAL["mfj"]) == 0
    assert tax.tax_from_brackets(-5_000, tax.FEDERAL["mfj"]) == 0
    assert tax.federal_ordinary(10_000, "mfj") == 0  # under the deduction


def test_known_bracket_values_mfj_2025():
    # taxable 100,000: 10%*23,850 + 12%*(96,950-23,850) + 22%*3,050
    expected = 0.10 * 23_850 + 0.12 * (96_950 - 23_850) + 0.22 * 3_050
    assert tax.tax_from_brackets(100_000, tax.FEDERAL["mfj"]) == pytest.approx(expected)
    # standard deduction applied on gross
    assert tax.federal_ordinary(131_500, "mfj") == pytest.approx(expected)


def test_bracket_continuity_and_marginal():
    b = tax.FEDERAL["mfj"]
    for lo, rate in b[1:]:
        below = tax.tax_from_brackets(lo - 1, b)
        above = tax.tax_from_brackets(lo + 1, b)
        # crossing a threshold changes tax smoothly (no jumps)
        assert above - below == pytest.approx(
            1 * rate + 1 * tax.marginal_rate(lo - 1, b), abs=0.01)
        assert tax.marginal_rate(lo + 1, b) == rate


def test_effective_below_marginal():
    for taxable in (50_000, 150_000, 400_000, 900_000):
        eff = tax.tax_from_brackets(taxable, tax.FEDERAL["mfj"]) / taxable
        assert eff < tax.marginal_rate(taxable, tax.FEDERAL["mfj"])


def test_ltcg_stacking():
    # ordinary 50k + gain 30k: whole gain inside the 0% band (< 96,700)
    assert tax.ltcg_tax(30_000, 50_000, "mfj") == 0
    # ordinary 90k + gain 30k: 6,700 at 0%, 23,300 at 15%
    assert tax.ltcg_tax(30_000, 90_000, "mfj") == pytest.approx(23_300 * 0.15)
    # deep in: all 15%
    assert tax.ltcg_tax(10_000, 200_000, "mfj") == pytest.approx(1_500)
    # gains never taxed at ordinary rates
    assert tax.ltcg_tax(100_000, 500_000, "mfj") <= 100_000 * 0.20 + 1e-9


def test_ma_flat_plus_surtax():
    assert tax.ma_tax(100_000) == pytest.approx(5_000)
    over = tax.MA_SURTAX_THRESHOLD + 100_000
    assert tax.ma_tax(over) == pytest.approx(over * 0.05 + 100_000 * 0.04)


def test_gross_up_nets_the_target():
    for net in (20_000, 90_000, 250_000):
        gross = tax.gross_up(net, other_income=50_000, filing="mfj")
        tax_due = (tax.total_ordinary(50_000 + gross, "mfj")
                   - tax.total_ordinary(50_000, "mfj"))
        assert gross - tax_due == pytest.approx(net, abs=1.0)


def test_gross_up_with_penalty():
    net = 50_000
    g0 = tax.gross_up(net, filing="mfj")
    g10 = tax.gross_up(net, filing="mfj", penalty=0.10)
    assert g10 > g0
    tax_due = tax.total_ordinary(g10, "mfj")
    assert g10 - tax_due - 0.10 * g10 == pytest.approx(net, abs=1.0)


def test_gross_up_zero():
    assert tax.gross_up(0) == 0.0


def test_ma_estate_tax_values():
    import numpy as np
    assert tax.ma_estate_tax(1_500_000) == 0
    assert tax.ma_estate_tax(2_000_000) == 0
    # hand-computed: credit-table tax(2.5M) 138,800 minus credit 99,600
    assert tax.ma_estate_tax(2_500_000) == pytest.approx(39_200)
    est = np.array([0.0, 2_000_000, 2_500_000, 10_000_000])
    out = tax.ma_estate_tax(est)
    assert out[0] == 0 and out[1] == 0
    assert out[2] == pytest.approx(39_200)
    assert np.all(np.diff(out) >= 0)  # monotone
    # marginal never exceeds the 16% top rate
    assert (tax.ma_estate_tax(10_100_000) - tax.ma_estate_tax(10_000_000)) <= 100_000 * 0.16 + 1


def test_eff_rate_helpers():
    # stacked rate exceeds from-zero rate
    assert tax.eff_rate_on(50_000, 300_000) > tax.eff_rate_on(50_000, 0)
    # bridge-style: gains on ~zero ordinary income pay MA only
    assert tax.cg_eff_rate_on(30_000, 0) == pytest.approx(0.05)
    assert tax.eff_rate_on(0) == 0.0


def test_ss_aware_grid_exact_against_brute_force():
    """The piecewise grid must match direct evaluation everywhere —
    including inside the 1.5x/1.85x SS phase-in ('tax torpedo')."""
    import numpy as np
    ss_tot = 51_000
    x, T = tax.ordinary_tax_grid(ss_total=ss_tot, ss_deflator=0.5)
    for w in np.linspace(0, 400_000, 481):
        fed_taxable = (w + tax.taxable_ss_of(w, ss_tot, deflator=0.5)
                       - tax.STANDARD_DEDUCTION["mfj"])
        brute = (tax.tax_from_brackets(fed_taxable, tax.FEDERAL["mfj"])
                 + tax.ma_tax(w))
        assert np.interp(w, x, T) == pytest.approx(brute, abs=0.5), w


def test_ss_torpedo_marginal_exceeds_bracket_rate():
    """Inside the phase-in, one more dollar of withdrawal drags $1.85
    of SS into taxation — effective marginal ~1.85x the bracket rate."""
    import numpy as np
    x, T = tax.ordinary_tax_grid(ss_total=40_000)
    w = 40_000  # prov 60k: deep in the 85% phase-in
    marg = (np.interp(w + 100, x, T) - np.interp(w, x, T)) / 100
    plain_x, plain_T = tax.ordinary_tax_grid(ss_total=0.0)
    plain = (np.interp(w + 100, plain_x, plain_T)
             - np.interp(w, plain_x, plain_T)) / 100
    assert marg > plain * 1.5


def test_ss_threshold_deflation_raises_tax():
    import numpy as np
    x1, T1 = tax.ordinary_tax_grid(ss_total=40_000, ss_deflator=1.0)
    x2, T2 = tax.ordinary_tax_grid(ss_total=40_000, ss_deflator=0.4)
    w = 30_000
    assert np.interp(w, x2, T2) >= np.interp(w, x1, T1)


def test_single_brackets_tighter_than_mfj():
    # the MFS/MFJ penalty math rests on this shape
    for taxable in (100_000, 250_000, 500_000):
        assert (tax.tax_from_brackets(taxable, tax.FEDERAL["single"])
                >= tax.tax_from_brackets(taxable, tax.FEDERAL["mfj"]))

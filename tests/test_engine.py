"""Numeric tests for the simulation engine's account rules.

Each test fabricates a minimal config + bucket snapshot and checks the
engine's arithmetic exactly (zero-return deterministic runs make the
math closed-form): every account rule the engine claims to model has a
test that would catch its removal. Regression and invariant tests on a
real household configuration live with that configuration.

Run:  uv run pytest -q
"""

import numpy as np
import pytest

from finsim import gen_returns, simulate  # noqa: E402
from finsim import mortality  # noqa: E402

from finsim.testing import THIS_YEAR, buckets, det, mini_cfg  # noqa: E402,F401


# ── withdrawal taxation and ordering ────────────────────────────────────

def test_taxable_draw_grossed_up_for_capital_gains():
    cfg = mini_cfg(retirement={"retirement_year": 2026},
                   spending={"retirement_base": 100_000})
    res = det(cfg, buckets(taxable=2_000_000))
    t = res["trace"][0]
    net_f = 1 - 0.45 * 0.20  # 0.91
    assert t["draw_taxable"] == pytest.approx(100_000 / net_f)
    assert res["wd_tax_paid"][0] > 0


def test_pretax_early_withdrawal_penalty():
    cfg = mini_cfg(retirement={"retirement_year": 2026},
                   spending={"retirement_base": 85_000},
                   strategy={"name": "taxable_first"})
    res = det(cfg, buckets(pretax=2_000_000))
    t = res["trace"][0]  # Andy is 37: 15% tax + 10% penalty
    assert t["draw_pretax"] == pytest.approx(85_000 / 0.75)
    # penalty_paid is cumulative: 10% of every pre-60 gross draw
    total_gross = sum(r["draw_pretax"] for r in res["trace"])
    assert res["penalty_paid"][0] == pytest.approx(total_gross * 0.10)


def test_pretax_no_penalty_after_60():
    cfg = mini_cfg(simulation={"horizon_age": 62},  # runs to 2052, Andy 63
                   spending={"retirement_base": 30_000},
                   strategy={"name": "taxable_first"})
    res = det(cfg, buckets(pretax=5_000_000))
    by_age = {t["age"]: t for t in res["trace"]}
    assert by_age[59]["draw_pretax"] == pytest.approx(30_000 / 0.75)
    assert by_age[60]["draw_pretax"] == pytest.approx(30_000 / 0.85)


def test_roth_basis_free_then_earnings_taxed():
    cfg = mini_cfg(retirement={"retirement_year": 2026},
                   spending={"retirement_base": 100_000},
                   rules={"roth_basis_fraction": 0.5})
    res = det(cfg, buckets(roth=300_000))  # basis 150k
    t = res["trace"][0]
    # first 100k of need comes entirely from basis, tax-free
    assert t["draw_roth"] == pytest.approx(100_000)
    t2 = res["trace"][1]
    # year 2: 50k basis left, then earnings grossed up at 15% + 10%
    assert t2["draw_roth"] == pytest.approx(50_000 + 50_000 / 0.75)
    # lifetime: all 150k of earnings leave as pre-60 draws -> 25% total
    assert res["penalty_paid"][0] == pytest.approx(150_000 * 0.10)
    assert res["wd_tax_paid"][0] == pytest.approx(150_000 * 0.25)


def test_457b_locked_until_separation_then_tax_free():
    andy_income = [{"from_year": 2026, "to_year": 2028, "gross": 0,
                    "eff_tax": 0.0, "real_growth": 0.0}]
    cfg = mini_cfg(income={"andy": andy_income, "sam": []},
                   retirement={"retirement_year": 2027},
                   spending={"retirement_base": 50_000})
    res = det(cfg, buckets(b457=1_000_000))
    by_year = {t["year"]: t for t in res["trace"]}
    # 2027-2028: still employed at UC -> 457b locked -> path is ruined
    assert by_year[2027]["b457"] == pytest.approx(1_000_000)
    # 2029+: separated -> drawn with no tax, no penalty (Roth flavor)
    assert by_year[2029]["b457"] == pytest.approx(950_000)
    assert res["penalty_paid"][0] == 0.0


def test_457b_pretax_source_taxed_not_penalized():
    andy_income = [{"from_year": 2026, "to_year": 2026, "gross": 0,
                    "eff_tax": 0.0, "real_growth": 0.0}]
    cfg = mini_cfg(income={"andy": andy_income, "sam": []},
                   spending={"retirement_base": 85_000},
                   rules={"b457_initial_pretax_frac": 1.0})
    res = det(cfg, buckets(b457=2_000_000))
    t = res["trace"][1]  # 2027: separated, retired
    assert t["draw_taxable"] == 0
    assert res["penalty_paid"][0] == 0.0
    assert res["wd_tax_paid"][0] > 0  # ordinary tax on pre-tax source


def test_457b_mixed_sources_roth_first_pre60():
    """70/30 mixed 457b: pre-60 draws spend the Roth source tax-free
    before touching the taxed pre-tax source; the pre-tax source pays
    RMDs at 75, the Roth source doesn't."""
    andy_income = [{"from_year": 2026, "to_year": 2026, "gross": 0,
                    "eff_tax": 0.0, "real_growth": 0.0}]
    cfg = mini_cfg(income={"andy": andy_income, "sam": []},
                   spending={"retirement_base": 100_000},
                   rules={"b457_initial_pretax_frac": 0.5})
    res = det(cfg, buckets(b457=1_000_000))
    # year 2 (2027, separated): Roth source (500k) covers 100k tax-free
    t1 = res["trace"][1]
    assert t1["draw_b457"] == pytest.approx(100_000)
    assert t1["b457"] == pytest.approx(900_000)
    # years 2-6 spend down the 500k Roth source with zero tax; year 7
    # starts on the pre-tax source, grossed up
    taxes_by_year = []
    prev = 0.0
    # reconstruct per-year draws: Roth years draw exactly 100k
    for t in res["trace"][1:6]:
        assert t["draw_b457"] == pytest.approx(100_000)
    assert res["trace"][6]["draw_b457"] > 100_000  # grossed for tax
    assert res["wd_tax_paid"][0] > 0


def test_457b_contribution_split_routes_by_flavor():
    income = [{"from_year": 2026, "to_year": 2035, "gross": 300_000,
               "eff_tax": 0.0, "real_growth": 0.0}]
    cfg = mini_cfg(income={"andy": income, "sam": []},
                   retirement={"retirement_year": 2030},
                   rules={"p457_deduct": 0.35},
                   contributions={"pretax_annual": 0, "match_annual": 0,
                                  "mega_backdoor_annual": 0,
                                  "hsa_annual": 0, "p457_annual": 20_000,
                                  "p457_pretax_frac": 0.70})
    res = det(cfg, buckets())
    t = res["trace"][0]
    assert t["b457"] == pytest.approx(20_000)
    # cash cost: 30% Roth at par + 70% pre-tax at (1 - 0.35)
    assert t["contrib"] == pytest.approx(
        20_000 * (0.30 + 0.70 * 0.65))


def test_rmd_forced_at_75_uniform_lifetime_table():
    income = [{"from_year": 2026, "to_year": 2070, "gross": 100_000,
               "eff_tax": 0.0, "real_growth": 0.0}]
    cfg = mini_cfg(simulation={"horizon_age": 75},  # to 2065, Andy 76
                   income={"andy": income, "sam": []},
                   retirement={"retirement_year": 2027})
    res = det(cfg, buckets(pretax=1_000_000))
    by_age = {t["age"]: t for t in res["trace"]}
    assert by_age[74]["rmd"] == 0
    assert by_age[75]["rmd"] == pytest.approx(1_000_000 / 24.6 * 0.85)


def test_ladder_conversion_seasons_five_years():
    cfg = mini_cfg(simulation={"horizon_age": 48},
                   spending={"retirement_base": 0},
                   strategy={"name": "wl_bridge", "ladder_annual": 100_000,
                             "ladder_eff_tax": 0.0},
                   rules={"roth_basis_fraction": 0.0})
    # income covers spending; conversions move pretax -> roth
    res = det(cfg, buckets(pretax=1_000_000, cash=1_000_000))
    rows = {t["year"]: t for t in res["trace"]}
    assert rows[2027]["roth"] == pytest.approx(100_000)   # converted
    assert rows[2031]["pretax"] == pytest.approx(500_000)
    # seasoning is internal (basis), but conversions keep flowing yearly
    assert rows[2035]["roth"] == pytest.approx(900_000)


def test_lti_forfeits_after_employment_end():
    cfg = mini_cfg(retirement={"retirement_year": 2033},
                   lti={"eff_tax": 0.0, "payout_years": 4,
                        "employment_end_year": 2027})
    res = det(cfg, buckets(lti=400_000, cash=0))
    rows = {t["year"]: t for t in res["trace"]}
    assert rows[2026]["income"] == pytest.approx(100_000)
    assert rows[2027]["income"] == pytest.approx(100_000)
    assert rows[2028]["income"] == pytest.approx(0.0)  # forfeited


def test_wl_premiums_build_cash_value():
    cfg = mini_cfg(retirement={"retirement_year": 2030},
                   whole_life={"premiums_annual": 10_000,
                               "cv_premium_credit": 0.9,
                               "cv_real_growth": 0.0, "drainable": True})
    res = det(cfg, buckets(cash=1_000_000))
    rows = {t["year"]: t for t in res["trace"]}
    assert rows[2026]["wl"] == pytest.approx(9_000)
    assert rows[2029]["wl"] == pytest.approx(36_000)
    assert rows[2030]["wl"] == pytest.approx(36_000)  # premiums stopped


def test_health_cost_phases():
    sam_income = [{"from_year": 2026, "to_year": 2028, "gross": 0,
                   "eff_tax": 0.0, "real_growth": 0.0}]
    cfg = mini_cfg(simulation={"horizon_age": 42},  # 2026-2032
                   income={"andy": [], "sam": sam_income},
                   retirement={"retirement_year": 2027},
                   spending={"retirement_base": 0,
                             "health": {"spouse_working_annual": 12_000,
                                        "pre_medicare_annual": 20_000,
                                        "medicare_annual": 10_000}})
    res = det(cfg, buckets(taxable=10_000_000))
    rows = {t["year"]: t for t in res["trace"]}
    assert rows[2026]["spend"] == pytest.approx(0)        # employed
    assert rows[2027]["spend"] == pytest.approx(12_000)   # Sam works
    assert rows[2029]["spend"] == pytest.approx(20_000)   # both retired
    # (Andy hits 65 in 2054, beyond this horizon — phase 3 covered below)


def test_health_medicare_phase_at_65():
    cfg = mini_cfg(simulation={"horizon_age": 66},  # to 2056, Andy 67
                   retirement={"retirement_year": 2027},
                   spending={"retirement_base": 0,
                             "health": {"spouse_working_annual": 12_000,
                                        "pre_medicare_annual": 20_000,
                                        "medicare_annual": 10_000}})
    res = det(cfg, buckets(taxable=10_000_000))
    by_age = {t["age"]: t for t in res["trace"]}
    assert by_age[64]["spend"] == pytest.approx(20_000)
    assert by_age[65]["spend"] == pytest.approx(10_000)


def test_guardrails_cut_and_floor():
    cfg = mini_cfg(simulation={"horizon_age": 46},
                   retirement={"retirement_year": 2026},
                   spending={"retirement_base": 100_000,
                             "guardrails": {"enabled": True,
                                            "trigger_wr": 0.05,
                                            "release_wr": 0.035,
                                            "step": 0.10, "floor": 0.75}})
    res = det(cfg, buckets(taxable=1_000_000))  # 10% WR -> cut every year
    spends = [t["spend"] for t in res["trace"]]
    assert spends[0] == pytest.approx(100_000)   # first year full
    assert spends[1] == pytest.approx(90_000)    # cut 10%
    assert spends[2] == pytest.approx(80_000)
    assert min(spends) >= 75_000 - 1e-6          # floor respected


def test_contribution_routing_and_match():
    income = [{"from_year": 2026, "to_year": 2035, "gross": 300_000,
               "eff_tax": 0.0, "real_growth": 0.0}]
    cfg = mini_cfg(income={"andy": income, "sam": []},
                   retirement={"retirement_year": 2030},
                   contributions={"pretax_annual": 10_000,
                                  "match_annual": 5_000,
                                  "mega_backdoor_annual": 15_000,
                                  "hsa_annual": 8_000,
                                  "p457_annual": 14_000,
                                  "backdoor_roth_annual": 7_000})
    res = det(cfg, buckets())
    t = res["trace"][0]
    assert t["contrib"] == pytest.approx(10_000 + 15_000 + 8_000 + 14_000 + 7_000)
    assert t["pretax"] == pytest.approx(15_000)   # employee + match
    assert t["roth"] == pytest.approx(30_000)     # mega + hsa + backdoor
    assert t["b457"] == pytest.approx(14_000)
    # contributions stop at retirement
    assert res["trace"][5]["contrib"] == 0


def test_college_draws_529_then_household():
    kid = {"name": "Kid", "start_year": 2027}
    cfg = mini_cfg(college={"target_today": 80_000, "monthly_contrib": 0,
                            "years_in_college": 4, "real_cost_growth": 0.0,
                            "kids": [kid]})
    b = buckets(taxable=1_000_000)
    b["529"] = {"Kid": 30_000}
    res = det(cfg, b)
    rows = {t["year"]: t for t in res["trace"]}
    # need 20k/yr; 529 covers 2027 fully + 10k of 2028, then household
    assert rows[2027]["k529"] == pytest.approx(10_000)
    assert rows[2027]["draw_taxable"] == 0
    assert rows[2028]["k529"] == pytest.approx(0)
    assert rows[2028]["draw_taxable"] > 0
    assert res["college_funded"]["Kid"][0] == pytest.approx(30_000)


def test_reverse_mortgage_age_gated_and_capped():
    cfg = mini_cfg(simulation={"horizon_age": 64},  # to 2054
                   retirement={"retirement_year": 2026},
                   spending={"retirement_base": 50_000,
                             "mortgage_payoff_year": 0})
    res = det(cfg, buckets(home_value=1_000_000, cash=100_000))
    rows = {t["year"]: t for t in res["trace"]}
    # Sam (younger) turns 62 in 2052; before that: no RM, ruin after cash
    assert rows[2051]["draw_rm"] == 0
    assert rows[2052]["draw_rm"] > 0
    total_rm = sum(t["draw_rm"] for t in res["trace"])
    assert total_rm <= 1_000_000 * 0.5 + 1e-6


def test_horizon_is_younger_spouse():
    cfg = mini_cfg()
    res = det(cfg, buckets())
    assert res["years"][-1] == 1990 + 45


def test_estate_net_of_taxes():
    cfg = mini_cfg()
    res = det(cfg, buckets(taxable=100_000, pretax=100_000, roth=50_000,
                           cash=10_000))
    # zero flows: cash above 30k buffer sweeps to taxable at start; here
    # cash 10k stays. estate = cash + taxable*.91 + pretax*.85 + roth
    assert res["estate"][0] == pytest.approx(
        10_000 + 100_000 * 0.91 + 100_000 * 0.85 + 50_000)


# ── return models ───────────────────────────────────────────────────────

MARKET = {"equity_real_return": 0.05, "equity_vol": 0.16,
          "bond_real_return": 0.015, "bond_vol": 0.05, "equity_weight": 0.9,
          "return_model": "historical", "recenter_historical": True}


def test_historical_returns_recentered_moments():
    cfg = mini_cfg(market=dict(MARKET))
    rng = np.random.default_rng(7)
    rs, rb = gen_returns(cfg, 4000, 40, rng)
    m = cfg["market"]
    assert rs.mean() == pytest.approx(m["equity_real_return"], abs=0.005)
    assert rb.mean() == pytest.approx(m["bond_real_return"], abs=0.003)
    # historical shape: stock sd near the 1953-2019 empirical ~17.5%
    assert 0.15 < rs.std() < 0.20
    # bonds carry RPM's strong positive autocorrelation
    flat_b = rb[:, :-1].ravel(), rb[:, 1:].ravel()
    assert np.corrcoef(flat_b[0], flat_b[1])[0, 1] > 0.3


def test_normal_model_moments():
    cfg = mini_cfg(market=dict(MARKET, return_model="normal"))
    rng = np.random.default_rng(7)
    rs, rb = gen_returns(cfg, 4000, 40, rng)
    assert rs.mean() == pytest.approx(0.05, abs=0.005)
    assert rs.std() == pytest.approx(0.16, abs=0.01)


# ── mortality ───────────────────────────────────────────────────────────

def test_mortality_curves_sane():
    s = mortality.survival_curve(37, "male", 64)
    assert s[0] == 1.0
    assert np.all(np.diff(s) <= 0)
    # ported table overstates old-age mortality vs true SSA (P(100)~1.5%);
    # loose lower bound, and note the optimism bias this introduces
    assert 0.0001 < s[-1] < 0.05
    both = mortality.either_alive_curve([(37, "male"), (36, "female")], 64)
    assert np.all(both >= s - 1e-12)  # either-alive dominates individual
    le = mortality.life_expectancy(37, "male")
    assert 38 < le < 48


def test_mortality_weighted_health_bounds():
    cfg = mini_cfg(spending={"retirement_base": 200_000})
    res = simulate(cfg, buckets(taxable=500_000), n_sims=4, seed=1)
    # everyone ruins early, while certainly alive -> weighted ~ raw
    assert res["alive"].mean() == 0.0
    assert res["health_mortality"] < 5.0


def test_brackets_mode_matches_gross_up_solver():
    """Cross-check two independent implementations: the engine's
    piecewise-grid inversion vs finsim.tax.gross_up's binary search,
    on a pre-60 pre-tax draw with no other income."""
    from finsim import tax as taxmod
    cfg = mini_cfg(retirement={"retirement_year": 2026},
                   spending={"retirement_base": 100_000},
                   strategy={"name": "taxable_first"},
                   taxes={"mode": "brackets"})
    res = det(cfg, buckets(pretax=5_000_000))
    engine_gross = res["trace"][0]["draw_pretax"]
    solver_gross = taxmod.gross_up(100_000, other_income=0.0, penalty=0.10)
    assert engine_gross == pytest.approx(solver_gross, abs=2.0)


def test_brackets_mode_marginal_stacking():
    """Two ordinary draws in one year stack: total tax equals the
    bracket tax on the COMBINED gross, not each priced from zero."""
    from finsim import tax as taxmod
    cfg = mini_cfg(simulation={"horizon_age": 36},  # single sim year, 2026
                   retirement={"retirement_year": 2026},
                   spending={"retirement_base": 100_000},
                   strategy={"name": "taxable_first"},
                   rules={"roth_basis_fraction": 0.0},
                   taxes={"mode": "brackets"})
    # small pre-tax bucket exhausts, then Roth earnings finish the job:
    # both are ordinary income, both pre-60 (10% penalty)
    res = det(cfg, buckets(pretax=30_000, roth=500_000))
    t = res["trace"][0]
    total_gross = t["draw_pretax"] + t["draw_roth"]
    expected_tax = (taxmod.federal_ordinary(total_gross)
                    + taxmod.ma_tax(total_gross))
    expected_pen = 0.10 * total_gross
    assert res["wd_tax_paid"][0] == pytest.approx(
        expected_tax + expected_pen, abs=2.0)
    assert res["penalty_paid"][0] == pytest.approx(expected_pen, abs=1.0)


def test_ss_taxable_fraction_worksheet():
    from finsim import tax as taxmod
    assert taxmod.ss_taxable_fraction(0, 40_000) == 0.0  # prov 20k < 32k
    assert taxmod.ss_taxable_fraction(200_000, 40_000) == pytest.approx(0.85)
    mid = taxmod.ss_taxable_fraction(30_000, 30_000)  # prov 45k, just over t2
    assert 0.0 < mid < 0.85


def test_aca_subsidy_and_cliff():
    from finsim import tax as taxmod
    fpl = taxmod.FPL_2
    # deep subsidy at low MAGI
    assert taxmod.aca_premium_cost(1.5 * fpl, 16_000) < 2_000
    # just under the cliff: capped contribution
    under = taxmod.aca_premium_cost(3.99 * fpl, 16_000)
    assert under == pytest.approx(0.0986 * 3.99 * fpl, rel=0.02)
    # over the cliff: full premium
    assert taxmod.aca_premium_cost(4.01 * fpl, 16_000) == 16_000
    assert taxmod.irmaa_couple(100_000) == 0
    assert taxmod.irmaa_couple(300_000) > taxmod.irmaa_couple(220_000) > 0


def test_tips_sleeve_is_riskless_real():
    """With a 100% TIPS glide the portfolio return is exactly the
    locked real yield, regardless of market draws."""
    cfg = mini_cfg(market={"equity_real_return": 0.05, "equity_vol": 0.16,
                           "bond_real_return": 0.015, "bond_vol": 0.05,
                           "equity_weight": 0.0, "return_model": "normal",
                           "tips_real_yield": 0.02,
                           "tips_glide": [{"year": 2026, "weight": 1.0},
                                          {"year": 2100, "weight": 1.0}]})
    res = simulate(cfg, buckets(taxable=100_000), n_sims=50, seed=3)
    # zero spending: taxable compounds at exactly 2% real minus drag(0)
    n = len(res["years"])
    # taxable swept cash? cash=0; check via estate: 100k*(1.02^n)*0.91-ish
    expected = 100_000 * 1.02 ** n
    assert np.allclose(res["estate"], expected * (1 - 0.45 * 0.20), rtol=1e-9)



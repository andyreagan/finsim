"""Synthetic-household builders for tests.

A minimal valid configuration (zero returns, no noise) and an empty
bucket snapshot, so a test can turn on exactly one rule and check the
engine's arithmetic in closed form. The package's own tests use these;
a private repo's invariant tests can import them too.

    from finsim.testing import mini_cfg, buckets, det
    cfg = mini_cfg(retirement={"retirement_year": 2026},
                   spending={"retirement_base": 50_000})
    res = det(cfg, buckets(pretax=1_000_000))

The two people are a fictional household ("andy" born 1989 is the
primary, "sam" born 1990 the spouse); income phases are keyed by those
names. Top-level dict overrides merge one level deep.
"""

from finsim.engine import simulate

THIS_YEAR = 2026  # the fixtures assume this is the current year


def mini_cfg(**over):
    """Minimal valid config: zero returns, one earner knob, no noise."""
    cfg = {
        "people": [
            {"name": "andy", "birth_year": 1989, "sex": "male",
             "ss_annual": 0, "ss_claim_age": 67},
            {"name": "sam", "birth_year": 1990, "sex": "female",
             "ss_annual": 0, "ss_claim_age": 67},
        ],
        "simulation": {"horizon_age": 45},  # horizon 2035, 10 sim years
        "market": {"equity_real_return": 0.0, "equity_vol": 0.0,
                   "bond_real_return": 0.0, "bond_vol": 0.0,
                   "equity_weight": 1.0, "return_model": "normal"},
        "income": {"andy": [], "sam": []},
        "spending": {"base": 0, "retirement_base": 0,
                     "mortgage_annual": 0, "mortgage_payoff_year": 0,
                     "student_loan_annual": 0, "student_loan_from": 1,
                     "student_loan_to": 0},
        "retirement": {"retirement_year": 2027},
        "college": {"target_today": 0, "monthly_contrib": 0,
                    "years_in_college": 4, "real_cost_growth": 0.0,
                    "kids": []},
        "whole_life": {"premiums_annual": 0, "cv_premium_credit": 0.0,
                       "cv_real_growth": 0.0, "drainable": True},
        "home": {"real_appreciation": 0.0, "reverse_mortgage_ltv": 0.5,
                 "selling_costs": 0.0},
        "lti": {"eff_tax": 0.35, "payout_years": 4},
        "taxes": {"pretax_withdrawal_eff": 0.15,
                  "taxable_gain_fraction": 0.45, "capital_gains_rate": 0.20,
                  "taxable_dividend_drag": 0.0},
        "rules": {"penalty_free_age": 60, "early_withdrawal_penalty": 0.10,
                  "roth_basis_fraction": 1.0, "rmd_age": 75,
                  "reverse_mortgage_min_age": 62},
        "contributions": {"pretax_annual": 0, "match_annual": 0,
                          "mega_backdoor_annual": 0, "hsa_annual": 0},
        "strategy": {"name": "wl_bridge", "ladder_annual": 0,
                     "ladder_eff_tax": 0.17},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def buckets(**over):
    """An all-zero bucket snapshot with the given buckets filled in."""
    b = {"cash": 0.0, "taxable": 0.0, "pretax": 0.0, "roth": 0.0,
         "hsa": 0.0, "b457": 0.0, "lti": 0.0, "whole_life": 0.0,
         "home_value": 0.0, "mortgage": 0.0, "student_loans": 0.0,
         "529": {}, "utma": 0.0}
    b.update(over)
    return b


def det(cfg, b):
    """One deterministic path at expected returns."""
    return simulate(cfg, b, n_sims=1, seed=1, deterministic=True)

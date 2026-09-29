#!/usr/bin/env python3
"""Monte Carlo goals simulator.

Reads assumptions from goals.toml, current balances from a
balances.json snapshot (or straight from a beancount ledger via the
accounts.toml registry), and simulates household cash flow year by
year until the planning horizon across many random return paths.

Account rules modeled:
  - contributions routed to their tax buckets while the primary earns
    (401k pre-tax + match, mega backdoor Roth, governmental 457b, HSA)
  - 10% early-withdrawal penalty on pre-tax (and Roth earnings) before
    penalty_free_age; Roth basis withdrawable free anytime
  - RMDs from pre-tax starting at rmd_age (SECURE 2.0 Uniform Lifetime Table)
  - reverse mortgage gated on the younger spouse reaching 62 and the
    mortgage being paid off
  - whole-life premiums stop at retirement (or premiums_to_year); cash
    value drainable tax-free (basis/loans)
  - optional Roth conversion ladder in low-income retirement years

Return models:
  - "historical": RPM-style (Bogleheads Retiree Portfolio Model) draws --
    autocorrelated percentile sequences mapped onto the empirical
    1953-2019 real-return distributions (fat tails + sequence risk),
    optionally recentered to the configured expected returns
  - "normal": i.i.d. normal draws

Usage (from the directory holding goals.toml + accounts.toml, or pass
--goals/--accounts/--ledger/--balances explicitly):
    finsim                            # base case
    finsim --scenario downshift
    finsim --strategy taxable_first
    finsim --deterministic            # expected-return trace
    finsim --sims 50000 --json
    finsim --dump-balances            # snapshot the ledger to balances.json
"""

import argparse
import copy
import json
import sys
import tomllib
from datetime import date
from pathlib import Path

import numpy as np

from finsim.ledger import load_ledger, to_usd  # noqa: E402
from finsim.mortality import either_alive_curve  # noqa: E402

HIST = Path(__file__).parent / "historical_returns.json"

# engine bucket types; the accounts.toml registry maps ledger accounts
# onto these ("excluded" is valid in the registry, never a bucket here)
BUCKET_SCALARS = ("cash", "taxable", "pretax", "roth", "hsa", "b457",
                  "lti", "whole_life", "home_value", "mortgage",
                  "student_loans", "utma")

# Withdrawal orders before/after penalty_free_age. Reverse mortgage is
# always the final backstop (age-gated separately).
STRATEGIES = {
    # drain taxable, then eat the pre-tax penalty if needed
    "taxable_first": {
        "pre": ["taxable", "pretax", "roth", "wl", "b457"],
        "post": ["taxable", "pretax", "roth", "wl", "b457"],
    },
    # bridge to 59.5 with the penalty-free assets in growth order:
    # taxable, then the 457b (no penalty after separation), then WL,
    # then Roth basis; conventional order after
    "wl_bridge": {
        "pre": ["taxable", "b457", "wl", "roth", "pretax"],
        "post": ["taxable", "pretax", "b457", "roth", "wl"],
    },
    # spend Roth basis before WL in the bridge years
    "roth_bridge": {
        "pre": ["taxable", "b457", "roth", "wl", "pretax"],
        "post": ["taxable", "pretax", "b457", "roth", "wl"],
    },
}

# IRS Uniform Lifetime Table (2022+), ages 73-100
ULT = {73: 26.5, 74: 25.5,
       75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2,
       81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
       87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8,
       93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3,
       99: 6.8, 100: 6.4}


# ── Ledger: current balances by tax bucket, per the accounts registry ──

def load_registry(path=None):
    """The accounts.toml registry: the executable account database."""
    return tomllib.loads(Path(path or "accounts.toml").read_text())["account"]


def get_buckets(main_bc: Path, registry=None, accounts=None):
    """Sum ledger balances into engine buckets as declared in
    accounts.toml (bucket + ledger_accounts [+ exclude, negate])."""
    from beancount.core import realization

    entries, price_map = load_ledger(main_bc)
    real_root = realization.realize(entries)

    def sum_tree(node):
        return to_usd(node.balance, price_map) + sum(sum_tree(c) for c in node.values())

    def node_at(path):
        node = real_root
        for part in path.split(":"):
            node = node.get(part)
            if node is None:
                return None
        return node

    def val(path):
        node = node_at(path)
        return sum_tree(node) if node is not None else 0.0

    registry = registry or load_registry(accounts)
    b = {k: 0.0 for k in BUCKET_SCALARS}
    b["529"] = {}
    for acct in registry:
        bucket = acct["bucket"]
        if bucket == "excluded":
            continue
        if bucket == "529":
            # one child bucket per kid, discovered from the ledger tree
            root = node_at(acct["ledger_accounts"][0])
            if root is not None:
                for child, node in root.items():
                    b["529"][child] = b["529"].get(child, 0.0) + sum_tree(node)
            continue
        total = sum(val(p) for p in acct["ledger_accounts"])
        total -= sum(val(p) for p in acct.get("exclude", []))
        if acct.get("negate"):
            total = -total
        b[bucket] += total
        # collectibles sleeve (28% federal rate) tracked for tax blending
        for p_ in acct.get("collectibles_accounts", []):
            b["metals"] = b.get("metals", 0.0) + val(p_)
    return b


# ── Config / scenarios ──────────────────────────────────────────────────

def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)  # lists (e.g. income phases) replace
    return out


def load_config(scenario: str | None, path=None):
    cfg = tomllib.loads(Path(path or "goals.toml").read_text())
    scenarios = cfg.pop("scenarios", {})
    if scenario:
        if scenario not in scenarios:
            sys.exit(f"Unknown scenario {scenario!r}. Available: {', '.join(scenarios)}")
        cfg = deep_merge(cfg, scenarios[scenario])
    return cfg, list(scenarios)


# ── Return generation ───────────────────────────────────────────────────

def _norm_cdf(z):
    """Standard normal CDF (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7)."""
    x = np.abs(z) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429))))
    erf = 1 - poly * np.exp(-x * x)
    return 0.5 * (1 + np.sign(z) * erf)


def _empirical_quantile(sorted_vals, u):
    """Map uniform draws u in [0,1] onto an empirical distribution."""
    idx = u * (len(sorted_vals) - 1)
    return np.interp(idx, np.arange(len(sorted_vals)), sorted_vals)


def gen_returns(cfg, n_sims, n_years, rng):
    """Per-asset (stock, bond) real returns per sim-year, per the model."""
    m = cfg["market"]
    model = m.get("return_model", "normal")

    if model == "normal":
        rs = rng.normal(m["equity_real_return"], m["equity_vol"], (n_sims, n_years))
        rb = rng.normal(m["bond_real_return"], m["bond_vol"], (n_sims, n_years))
        return rs, rb

    if model != "historical":
        sys.exit(f"Unknown return_model {model!r}")

    hist = json.loads(HIST.read_text())
    stocks = np.array(hist["stocks"]) / 100.0
    bonds = np.array(hist["bonds"]) / 100.0
    if m.get("recenter_historical", True):
        stocks = stocks + (m["equity_real_return"] - stocks.mean())
        bonds = bonds + (m["bond_real_return"] - bonds.mean())
    stocks_sorted, bonds_sorted = np.sort(stocks), np.sort(bonds)

    # RPM-style: per-asset AR(1) percentile sequences (sequence-of-returns
    # risk), cross-correlated stock/bond draws, historical shape.
    rho_s, rho_b = 0.171, 0.584   # RPM Monte Carlo Setup defaults
    cross = float(np.corrcoef(stocks, bonds)[0, 1])

    zs = np.empty((n_sims, n_years))
    zb = np.empty((n_sims, n_years))
    eps_s = rng.standard_normal((n_sims, n_years))
    eps_b_ind = rng.standard_normal((n_sims, n_years))
    eps_b = cross * eps_s + np.sqrt(1 - cross**2) * eps_b_ind
    zs[:, 0], zb[:, 0] = eps_s[:, 0], eps_b[:, 0]
    for t in range(1, n_years):
        zs[:, t] = rho_s * zs[:, t - 1] + np.sqrt(1 - rho_s**2) * eps_s[:, t]
        zb[:, t] = rho_b * zb[:, t - 1] + np.sqrt(1 - rho_b**2) * eps_b[:, t]

    us, ub = _norm_cdf(zs), _norm_cdf(zb)
    rs = _empirical_quantile(stocks_sorted, us)
    rb = _empirical_quantile(bonds_sorted, ub)
    return rs, rb


# ── Simulation ──────────────────────────────────────────────────────────

def simulate(cfg, buckets, n_sims, seed, strategy=None, deterministic=False,
             trace=False, returns=None):
    """returns: optional (n_sims, n_years) blended real-return matrix,
    overriding the return model — for tests and external return feeds."""
    rng = np.random.default_rng(seed)
    this_year = date.today().year

    p = cfg["people"]
    # until the YOUNGER (later-born) spouse reaches horizon_age
    people = p                      # list of {name, birth_year, sex, ss_*}
    primary = people[0]             # owns the retirement accounts by default
    youngest_by = max(pp["birth_year"] for pp in people)
    horizon_age = cfg["simulation"]["horizon_age"]
    horizon_year = youngest_by + horizon_age
    years = np.arange(this_year, horizon_year + 1)
    n_years = len(years)

    m = cfg["market"]
    # equity weight per year: optional glide (bond tent), else constant
    glide = m.get("equity_glide")
    if glide:
        w_t = np.interp(years, [g["year"] for g in glide],
                        [g["weight"] for g in glide])
    else:
        w_t = np.full(n_years, float(m["equity_weight"]))
    # optional TIPS-ladder sleeve: a riskless REAL asset (held-to-
    # maturity ladder = locked real yield, zero vol) the stock/bond
    # model can't express; takes its weight out of the bond share
    tips_glide = m.get("tips_glide")
    if tips_glide:
        wtips_t = np.interp(years, [g["year"] for g in tips_glide],
                            [g["weight"] for g in tips_glide])
    else:
        wtips_t = np.zeros(n_years)
    tips_yield = m.get("tips_real_yield", 0.02)
    # tips crowd out bonds first, then equity — weights always sum to 1
    w_t = np.minimum(w_t, 1.0 - wtips_t)
    wbond_t = np.maximum(0.0, 1.0 - w_t - wtips_t)
    if returns is not None:
        returns = np.asarray(returns, dtype=float)
        assert returns.shape == (n_sims, n_years)
    else:
        if deterministic:
            n_sims = 1
            rs = np.full((1, n_years), m["equity_real_return"])
            rb = np.full((1, n_years), m["bond_real_return"])
            trace = True
        else:
            rs, rb = gen_returns(cfg, n_sims, n_years, rng)
        returns = (w_t[None, :] * rs + wbond_t[None, :] * rb
                   + wtips_t[None, :] * tips_yield)

    tx = cfg["taxes"]
    rules = cfg["rules"]
    strat_cfg = cfg["strategy"]
    tax_mode = tx.get("mode", "effective")
    strat_name = strategy or strat_cfg["name"]
    order = STRATEGIES[strat_name]
    ladder_annual = strat_cfg.get("ladder_annual", 0)
    ladder_eff = strat_cfg.get("ladder_eff_tax", 0.12)
    penalty = rules["early_withdrawal_penalty"]
    pf_age = rules["penalty_free_age"]

    # Withdrawal taxes. "effective" mode: flat config rates (what the
    # unit tests pin). "brackets" mode: EXACT per-sim progressive math —
    # each year gets a piecewise-linear composite tax function T(t)
    # (fed 2025 + MA, with taxable SS carved out of the MA base via the
    # provisional-income worksheet), and every ordinary-income event
    # (RMDs, ladder conversions, pre-tax/457b/Roth-earnings draws)
    # taxes its slice via np.interp on the grid, stacking on a per-sim
    # running ordinary-income level. Gross-from-net inverts the same
    # piecewise function, so the marginal rate moves within a single
    # withdrawal. Cap gains use a per-year effective rate (stacking
    # approximation) + NIIT when expected MAGI clears 250k.
    ret_year = cfg["retirement"]["retirement_year"]
    # SS lives on each person (ss_annual, ss_claim_age); optional global
    # toggle for scenarios
    ss_enabled = cfg.get("social_security", {}).get("enabled", True)
    filing = tx.get("filing", "mfj")
    state = tx.get("state", "MA")
    # deterministic LTI settlement schedule (needed by the tax anchor)
    _lti_end = min(cfg["lti"].get("employment_end_year", ret_year - 1),
                   ret_year - 1)
    _lti_annual = buckets["lti"] / cfg["lti"]["payout_years"]
    lti_sched = {}
    _rem = buckets["lti"]
    for _yr in years:
        if _yr <= _lti_end and _rem > 0:
            lti_sched[int(_yr)] = min(_rem, _lti_annual)
            _rem -= lti_sched[int(_yr)]

    MEDICARE_ON_WAGES = 0.0235  # 1.45% + 0.9% additional (>250k MFJ)

    if tax_mode == "brackets":
        from finsim import tax as taxmod
        ref = tx.get("bracket_ref_withdrawal", 75_000)
        conv_ref = strat_cfg.get("ladder_annual", 0) or ref
        metals_share = (buckets.get("metals", 0.0) / buckets["taxable"]
                        if buckets["taxable"] > 0 else 0.0)
        pretax_rate_t = np.empty(n_years)   # informational eff rates
        cg_rate_t = np.empty(n_years)
        ladder_rate_t = np.empty(n_years)
        lti_rate_t = np.full(n_years, cfg["lti"]["eff_tax"])
        year_tax = []
        _contrib = cfg["contributions"]
        _contrib = [_contrib] if isinstance(_contrib, dict) else _contrib
        for i, yr in enumerate(years):
            wages = 0.0
            for pp in people:
                for ph in cfg["income"].get(pp["name"], []):
                    if ph["from_year"] <= yr <= ph["to_year"]:
                        wages += ph["gross"] * (1 + ph["real_growth"]) ** (
                            yr - ph["from_year"])
            if yr < ret_year:
                for cp in _contrib:
                    if cp.get("from_year", 0) <= yr <= cp.get("to_year", 9999):
                        wages -= cp["pretax_annual"]
                        wages -= cp["hsa_annual"]  # payroll HSA is pre-tax
                        wages -= (cp.get("p457_annual", 0)
                                  * cp.get("p457_pretax_frac", 0.0))
            wages = max(0.0, wages)
            ss_tot = 0.0
            if ss_enabled:
                for pp in people:
                    if yr - pp["birth_year"] >= pp.get("ss_claim_age", 67):
                        ss_tot += pp.get("ss_annual", 0)
            # SS-aware grid: taxable SS is part of the piecewise tax
            # function itself (exact phase-in slopes), with the frozen
            # nominal 32k/44k thresholds deflated into real dollars
            deflator = 1.0 / (1 + tx.get("cpi", 0.025)) ** (yr - years[0])
            gx, gT = taxmod.ordinary_tax_grid(
                ss_total=ss_tot, ss_deflator=deflator,
                filing=filing, state=state)
            lti_g = lti_sched.get(int(yr), 0.0)
            o0 = wages + lti_g  # withdrawals stack ABOVE LTI settlements
            year_tax.append({"x": gx, "T": gT, "o0": o0, "ss_tot": ss_tot})
            if lti_g > 0:
                lti_rate_t[i] = ((np.interp(wages + lti_g, gx, gT)
                                  - np.interp(wages, gx, gT)) / lti_g
                                 + MEDICARE_ON_WAGES)

            def _T(v):
                return float(np.interp(v, gx, gT))
            pretax_rate_t[i] = (_T(o0 + ref) - _T(o0)) / ref
            ladder_rate_t[i] = (_T(o0 + conv_ref) - _T(o0)) / conv_ref
            tss = taxmod.taxable_ss_of(o0 + ref, ss_tot, filing=filing,
                                       deflator=deflator)
            ord_taxable = max(
                0.0, o0 + tss - taxmod.STANDARD_DEDUCTION[filing])
            gain_ref = ref * tx["taxable_gain_fraction"]
            cg_rate_t[i] = taxmod.cg_eff_rate_on(
                gain_ref, ord_taxable, filing=filing, state=state)
            if o0 + ref + gain_ref > 250_000:  # NIIT on investment income
                cg_rate_t[i] += 0.038
            # metals sleeve of the taxable bucket pays the 28%
            # collectibles rate (+MA) instead of LTCG
            if metals_share > 0:
                cg_rate_t[i] = ((1 - metals_share) * cg_rate_t[i]
                                + metals_share * (0.28 + taxmod.MA_RATE))
    else:
        pretax_rate_t = np.full(n_years, tx["pretax_withdrawal_eff"])
        cg_rate_t = np.full(n_years, tx["capital_gains_rate"])
        ladder_rate_t = np.full(
            n_years, strat_cfg.get("ladder_eff_tax", 0.17))
        lti_rate_t = np.full(n_years, cfg["lti"]["eff_tax"])
        year_tax = None

    # per-year working values (bucket_draw reads these via closure)
    taxable_tax_rate = tx["taxable_gain_fraction"] * cg_rate_t[0]
    pretax_eff = pretax_rate_t[0]
    ord_level = np.zeros(n_sims)  # per-sim ordinary income this year
    cur_gx = cur_gT = None        # this year's tax grid (brackets mode)

    def tax_on_ordinary(amount):
        """Tax on `amount` of ordinary income stacked on ord_level;
        advances ord_level. Vectorized."""
        nonlocal ord_level
        if year_tax is None:
            return amount * pretax_eff  # flat effective mode
        t = (np.interp(ord_level + amount, cur_gx, cur_gT)
             - np.interp(ord_level, cur_gx, cur_gT))
        ord_level = ord_level + amount
        return t

    def gross_for_net(deficit, pen, rate_flat):
        """Gross ordinary withdrawal so that gross - tax - pen*gross
        nets `deficit`, at the current per-sim ord_level. Exact
        piecewise-linear inversion in brackets mode."""
        if year_tax is None:
            return deficit / (1 - rate_flat - pen)
        Hx = cur_gx * (1 - pen) - cur_gT   # net-of-tax-and-penalty
        H0 = np.interp(ord_level, cur_gx, Hx)
        total = np.interp(H0 + deficit, Hx, cur_gx)
        return np.maximum(0.0, total - ord_level)

    # contributions: a single table, or a list of phases with
    # from_year/to_year (unbounded phases run until retirement)
    contrib_phases = cfg["contributions"]
    if isinstance(contrib_phases, dict):
        contrib_phases = [contrib_phases]

    # per-sim state (real dollars)
    cash = np.full(n_sims, buckets["cash"], float)
    taxable = np.full(n_sims, buckets["taxable"], float)
    pretax = np.full(n_sims, buckets["pretax"], float)
    roth = np.full(n_sims, buckets["roth"] + buckets["hsa"], float)
    # basis withdrawable penalty-free before 59.5: a fraction of today's
    # Roth plus the whole HSA (assumed to cover medical spend)
    roth_basis = np.full(
        n_sims, rules["roth_basis_fraction"] * buckets["roth"] + buckets["hsa"], float
    )
    # 457(b) split by source: pre-tax (taxed on withdrawal, has RMDs)
    # vs Roth (tax-free out); today's balance flavor set by
    # rules.b457_initial_pretax_frac (0 = all Roth, the current reality)
    _f0 = rules.get("b457_initial_pretax_frac", 0.0)
    b457p = np.full(n_sims, buckets["b457"] * _f0, float)
    b457r = np.full(n_sims, buckets["b457"] * (1 - _f0), float)
    wl = np.full(n_sims, buckets["whole_life"], float)
    home = np.full(n_sims, buckets["home_value"], float)
    rm_drawn = np.zeros(n_sims)  # reverse-mortgage principal drawn
    lti_remaining = np.full(n_sims, buckets["lti"], float)
    k529 = {k: np.full(n_sims, v, float) for k, v in buckets["529"].items()}
    pending_conv = []  # (year, amount) ladder conversions awaiting 5-yr seasoning
    conv_unseasoned = np.zeros(n_sims)  # conversions <5yr old: penalty-only if drawn
    prev_taxable_gross = np.zeros(n_sims)  # last year's taxable-bucket sales (ACA MAGI)
    wd_tax_paid = np.zeros(n_sims)   # cumulative withdrawal taxes + penalties
    penalty_paid = np.zeros(n_sims)  # cumulative early-withdrawal penalties only

    alive = np.ones(n_sims, bool)  # not ruined
    ruin_year = np.full(n_sims, 0, int)

    coll = cfg["college"]
    kids = coll["kids"]
    college_funded = {k["name"]: np.zeros(n_sims) for k in kids}  # $ paid from 529
    college_need_per_kid = {}
    college_at_start = {}  # name -> (balance at start, total need at start)

    sp = cfg["spending"]
    wl_cfg = cfg["whole_life"]
    home_cfg = cfg["home"]
    lti_cfg = cfg["lti"]
    lti_annual_gross = buckets["lti"] / lti_cfg["payout_years"]
    # tranches settle only while employed at MM; the rest forfeits
    lti_end = min(lti_cfg.get("employment_end_year", ret_year - 1), ret_year - 1)
    wl_premiums_to = wl_cfg.get("premiums_to_year", ret_year - 1)
    # last working year among the NON-primary people (health phases)
    others_last_year = max(
        (ph["to_year"] for pp in people[1:]
         for ph in cfg["income"].get(pp["name"], [])), default=0
    )
    # 457b unlocks when the primary separates from the sponsor (last year of any income)
    primary_last_year = max(
        (ph["to_year"] for ph in cfg["income"].get(primary["name"], [])),
        default=0
    )
    p457_deduct = rules.get("p457_deduct", 0.0)
    health = sp.get("health")
    guard = sp.get("guardrails", {})
    gmult = np.ones(n_sims)   # guardrails spending multiplier
    gk_w0 = None              # GK initial withdrawal rate, set at retirement
    prev_return = np.zeros(n_sims)
    magi_lag = [None, None]   # MAGI(t-1), MAGI(t-2) for IRMAA's 2-yr lag

    mortgage_bal = buckets["mortgage"]

    # yearly percentile tracking of total net worth (incl. home equity)
    nw_p10 = np.zeros(n_years)
    nw_p50 = np.zeros(n_years)
    nw_p90 = np.zeros(n_years)
    trace_rows = [] if trace else None

    def bucket_draw(name, deficit, primary_age):
        """Draw from one bucket to cover deficit; returns net cash raised."""
        nonlocal taxable, pretax, roth, roth_basis, wl, b457p, b457r, conv_unseasoned
        raised = np.zeros(n_sims)
        gross_taken = np.zeros(n_sims)
        if name == "taxable":
            net_f = 1 - taxable_tax_rate
            gross = np.minimum(taxable, deficit / net_f)
            taxable -= gross
            raised = gross * net_f
            wd_tax_paid[:] += gross * taxable_tax_rate
            gross_taken = gross
        elif name == "pretax":
            pen = penalty if primary_age < pf_age else 0.0
            gross = np.minimum(pretax, gross_for_net(deficit, pen, pretax_eff))
            pretax -= gross
            t = tax_on_ordinary(gross)
            raised = gross - t - gross * pen
            wd_tax_paid[:] += t + gross * pen
            penalty_paid[:] += gross * pen
            gross_taken = gross
        elif name == "roth":
            if primary_age >= pf_age:
                gross = np.minimum(roth, deficit)
                roth -= gross
                roth_basis[:] = np.minimum(roth_basis, roth)
                raised = gross
                gross_taken = gross
            else:
                # basis first, tax/penalty-free
                free = np.minimum(np.minimum(roth_basis, roth), deficit)
                roth -= free
                roth_basis -= free
                remaining = deficit - free
                # unseasoned conversions: 10% penalty only (their income
                # tax was paid at conversion)
                uns = np.minimum(np.minimum(conv_unseasoned, roth),
                                 remaining / (1 - penalty))
                roth -= uns
                conv_unseasoned -= uns
                wd_tax_paid[:] += uns * penalty
                penalty_paid[:] += uns * penalty
                remaining = np.maximum(0.0, remaining - uns * (1 - penalty))
                # then earnings: ordinary income + 10% penalty
                gross = np.minimum(
                    roth, gross_for_net(remaining, penalty, pretax_eff))
                roth -= gross
                t = tax_on_ordinary(gross)
                wd_tax_paid[:] += t + gross * penalty
                penalty_paid[:] += gross * penalty
                raised = free + uns * (1 - penalty) + gross - t - gross * penalty
                gross_taken = free + uns + gross
        elif name == "wl":
            if wl_cfg["drainable"]:
                gross = np.minimum(wl, deficit)
                wl -= gross
                raised = gross
                gross_taken = gross
        elif name == "b457":
            # governmental 457b: no penalty ever, but locked until the
            # primary separates. Roth source tax-free; pre-tax source
            # ordinary income. Pre-60: spend Roth first; post-60:
            # pre-tax first (burn the taxed source before RMD era).
            if yr > primary_last_year:
                first, second = ((b457r, b457p) if primary_age < pf_age
                                 else (b457p, b457r))
                takes = []
                for src in (first, second):
                    if src is b457r:
                        g = np.minimum(src, deficit)
                        t = np.zeros(n_sims)
                        net = g
                    else:
                        g = np.minimum(
                            src, gross_for_net(deficit, 0.0, pretax_eff))
                        t = tax_on_ordinary(g)
                        net = g - t
                    src -= g
                    deficit = np.maximum(0.0, deficit - net)
                    wd_tax_paid[:] += t
                    takes.append((g, net))
                gross_taken = takes[0][0] + takes[1][0]
                raised = takes[0][1] + takes[1][1]
        return raised, gross_taken

    for i, yr in enumerate(years):
        r = returns[:, i]
        primary_age = yr - primary["birth_year"]
        min_age = yr - youngest_by
        # this year's withdrawal tax machinery (bucket_draw reads these)
        pretax_eff = pretax_rate_t[i]
        taxable_tax_rate = tx["taxable_gain_fraction"] * cg_rate_t[i]
        ladder_eff = ladder_rate_t[i]
        if year_tax is not None:
            yt = year_tax[i]
            cur_gx, cur_gT = yt["x"], yt["T"]
            ord_level = np.full(n_sims, yt["o0"])
        else:
            yt = None
        # grow market buckets (taxable also pays annual dividend tax)
        taxable *= 1 + r - tx.get("taxable_dividend_drag", 0.0)
        for arr in (pretax, roth, b457p, b457r):
            arr *= 1 + r
        for k in k529:
            k529[k] *= 1 + r
        wl *= 1 + wl_cfg["cv_real_growth"]
        home *= 1 + home_cfg["real_appreciation"]

        # income
        net_income = np.zeros(n_sims)
        for pp in people:
            for ph in cfg["income"].get(pp["name"], []):
                if ph["from_year"] <= yr <= ph["to_year"]:
                    g = ph["gross"] * (1 + ph["real_growth"]) ** (yr - ph["from_year"])
                    net_income += g * (1 - ph["eff_tax"])
        # LTI settles over payout window, only while still employed at MM
        if yr <= lti_end:
            settle = np.minimum(lti_remaining, lti_annual_gross)
            net_income += settle * (1 - lti_rate_t[i])
            lti_remaining -= settle
        # social security
        ss_income = 0.0
        if ss_enabled:
            for pp in people:
                if yr - pp["birth_year"] >= pp.get("ss_claim_age", 67):
                    ss_income += pp.get("ss_annual", 0)
        net_income += ss_income

        # retirement-account contributions while the primary works: diverted from
        # cash flow into their buckets; match is employer money on top
        contrib_out = 0.0
        if yr < ret_year:
            for cp in contrib_phases:
                if not cp.get("from_year", 0) <= yr <= cp.get("to_year", 9999):
                    continue
                p457 = cp.get("p457_annual", 0)
                pfrac = cp.get("p457_pretax_frac", 0.0)
                bdr = cp.get("backdoor_roth_annual", 0)
                contrib_out += (
                    cp["pretax_annual"]
                    + cp["mega_backdoor_annual"]
                    + cp["hsa_annual"]
                    # pre-tax fraction costs less cash (deduction)
                    + p457 * (1 - pfrac * p457_deduct)
                    + bdr
                )
                pretax += cp["pretax_annual"] + cp["match_annual"]
                b457p += p457 * pfrac
                b457r += p457 * (1 - pfrac)
                roth += cp["mega_backdoor_annual"] + cp["hsa_annual"] + bdr
                roth_basis += cp["mega_backdoor_annual"] + cp["hsa_annual"] + bdr

        # RMDs from pre-tax starting at rmd_age (before spending so the
        # forced income feeds IRMAA)
        rmd_net = np.zeros(n_sims)
        if primary_age >= rules["rmd_age"]:
            div = ULT[min(max(primary_age, min(ULT)), 100)]
            rmd = pretax / div
            pretax -= rmd
            t = tax_on_ordinary(rmd)
            rmd_net = rmd - t
            wd_tax_paid[:] += t
            cash += rmd_net
            rmd457 = b457p / div  # pre-tax 457 source has RMDs too
            b457p -= rmd457
            t = tax_on_ordinary(rmd457)
            cash += rmd457 - t
            wd_tax_paid[:] += t
            rmd_net = rmd_net + rmd457 - t

        # Roth conversion ladder (before spending: conversions raise
        # MAGI and price into ACA premiums); converted principal only
        # becomes withdrawable basis after the 5-year rule
        while pending_conv and pending_conv[0][0] <= yr - 5:
            # season what hasn't already been withdrawn
            entry = np.minimum(pending_conv.pop(0)[1], conv_unseasoned)
            roth_basis += entry
            conv_unseasoned -= entry
        ladder_conv = 0.0
        if ladder_annual and ret_year <= yr and primary_age < rules["rmd_age"]:
            conv = np.minimum(pretax, ladder_annual)
            pretax -= conv
            roth += conv
            conv_unseasoned = conv_unseasoned + conv
            pending_conv.append((yr, conv))
            tax = tax_on_ordinary(conv) if year_tax is not None \
                else conv * ladder_eff
            cash -= tax
            wd_tax_paid[:] += tax
            ladder_conv = float(np.median(conv))

        # spending (guardrails multiplier applies to the retirement base)
        if yr >= ret_year:
            spend = sp["retirement_base"] * gmult.copy()
        else:
            spend = np.full(n_sims, float(sp["base"]))
        # health insurance once employer coverage ends (while the primary works
        # it's covered inside base). In brackets mode the ACA years are
        # MAGI-aware (subsidy below 400% FPL, cliff above — conversions
        # and last year's realized gains price in), and Medicare years
        # add IRMAA.
        if health and yr >= ret_year:
            if yr <= others_last_year:
                spend += health["spouse_working_annual"]
            elif primary_age < 65:
                if year_tax is not None and health.get("aca", False):
                    from finsim import tax as taxmod
                    magi = (ord_level + yt["ss_tot"]
                            + prev_taxable_gross * tx["taxable_gain_fraction"])
                    premium = min(health.get("aca_premium_portion", 16_000),
                                  health["pre_medicare_annual"])
                    oop = health["pre_medicare_annual"] - premium
                    spend += taxmod.aca_premium_cost(magi, premium) + oop
                else:
                    spend += health["pre_medicare_annual"]
            else:
                spend += health["medicare_annual"]
                if year_tax is not None and health.get("aca", False):
                    from finsim import tax as taxmod
                    magi_now = (ord_level + yt["ss_tot"]
                                + prev_taxable_gross
                                * tx["taxable_gain_fraction"])
                    # IRMAA reads MAGI from two years back
                    magi = magi_lag[1] if magi_lag[1] is not None else magi_now
                    spend += taxmod.irmaa_couple(magi)
        if yr <= sp["mortgage_payoff_year"]:
            spend += sp["mortgage_annual"]
            mortgage_bal = max(0.0, mortgage_bal - 0.55 * sp["mortgage_annual"])
        if sp["student_loan_from"] <= yr <= sp["student_loan_to"]:
            spend += sp["student_loan_annual"]
        for ot in sp.get("one_time", []):
            if yr == ot["year"]:
                spend += ot["amount"]
        if yr <= wl_premiums_to and yr < ret_year:
            spend += wl_cfg["premiums_annual"]
            # most of the premium builds cash value (ledger-calibrated)
            wl += wl_cfg["premiums_annual"] * wl_cfg.get("cv_premium_credit", 0.0)

        # college: contributions until start; needs during window from 529
        college_out = np.zeros(n_sims)
        for kid in kids:
            name, start = kid["name"], kid["start_year"]
            if yr < start:
                k529[name] += coll["monthly_contrib"] * 12
                spend += coll["monthly_contrib"] * 12
            elif start <= yr < start + coll["years_in_college"]:
                if yr == start and name not in college_at_start:
                    total_need = sum(
                        coll["target_today"] / coll["years_in_college"]
                        * (1 + coll["real_cost_growth"]) ** (start + j - this_year)
                        for j in range(coll["years_in_college"])
                    )
                    college_at_start[name] = (k529[name].copy(), total_need)
                need = (
                    coll["target_today"]
                    / coll["years_in_college"]
                    * (1 + coll["real_cost_growth"]) ** (yr - this_year)
                )
                college_need_per_kid[name] = (
                    college_need_per_kid.get(name, 0.0) + need
                )
                from_529 = np.minimum(k529[name], need)
                k529[name] -= from_529
                college_funded[name] += from_529
                shortfall = need - from_529
                spend += shortfall  # shortfall hits household cash flow
                college_out += shortfall

        # guardrails: observe this year's withdrawal pressure, adjust the
        # spending multiplier for NEXT year (cut in bad stretches,
        # restore when the portfolio recovers)
        if guard.get("enabled") and yr >= ret_year:
            liquid = cash + taxable + pretax + roth + b457p + b457r + wl
            need = np.maximum(0, spend - net_income)
            wr = need / np.maximum(liquid, 1.0)
            if guard.get("style", "simple") == "gk":
                # Guyton-Klinger (2006): ratio rules against the initial
                # withdrawal rate, per-sim, captured at retirement
                if gk_w0 is None:
                    gk_w0 = np.maximum(wr, 1e-6)
                ratio = wr / gk_w0
                step = guard.get("adjust", 0.10)
                gmult = np.where(
                    ratio > guard.get("upper", 1.20),          # capital
                    np.maximum(gmult * (1 - step),             # preservation
                               guard.get("floor", 0.75)),
                    np.where(ratio < guard.get("lower", 0.80),  # prosperity
                            np.minimum(gmult * (1 + step),
                                       guard.get("ceiling", 1.50)), gmult))
                if guard.get("inflation_rule", True):
                    # skip the inflation adjustment after a down year: in
                    # real terms, an unrecovered cut of ~cpi
                    cpi = tx.get("cpi", 0.025)
                    gmult = np.where(prev_return < 0, gmult / (1 + cpi),
                                     gmult)
                gmult = np.maximum(gmult, guard.get("floor", 0.75))
            else:
                gmult = np.where(
                    wr > guard["trigger_wr"],
                    np.maximum(gmult - guard["step"], guard["floor"]),
                    np.where(wr < guard["release_wr"],
                             np.minimum(gmult + guard["step"], 1.0), gmult),
                )

        # net household cash flow
        flow = net_income - spend - contrib_out
        cash += flow

        # withdraw in strategy order to bring cash back to >= 0
        deficit = np.maximum(0, -cash)
        draws = {}
        for bname in (order["pre"] if primary_age < pf_age else order["post"]):
            if not deficit.any():
                break
            raised, gross_taken = bucket_draw(bname, deficit, primary_age)
            cash += raised
            draws[bname] = gross_taken
            deficit = np.maximum(0, -cash)
        # reverse mortgage: final backstop, age- and mortgage-gated
        if (
            yr > sp["mortgage_payoff_year"]
            and min_age >= rules["reverse_mortgage_min_age"]
            and deficit.any()
        ):
            capacity = np.maximum(0, home * home_cfg["reverse_mortgage_ltv"] - rm_drawn)
            take = np.minimum(capacity, deficit)
            rm_drawn += take
            cash += take
            draws["rm"] = take
            deficit = np.maximum(0, -cash)

        # ruin: still short after all backstops
        newly_ruined = alive & (deficit > 1e-6)
        ruin_year[newly_ruined] = yr
        alive &= ~newly_ruined
        cash = np.maximum(cash, 0)  # ruined paths pinned; alive paths already >= 0

        prev_taxable_gross = draws.get("taxable", np.zeros(n_sims))
        prev_return = r
        if year_tax is not None:
            magi_lag = [ord_level + yt["ss_tot"]
                        + prev_taxable_gross * tx["taxable_gain_fraction"],
                        magi_lag[0]]

        # sweep excess cash above a buffer into taxable
        buffer = 30000
        excess = np.maximum(0, cash - buffer)
        cash -= excess
        taxable += excess

        nw = (
            cash + taxable + pretax + roth + b457p + b457r + wl
            + home * (1 - home_cfg["selling_costs"]) - rm_drawn - mortgage_bal
        )
        nw_p10[i], nw_p50[i], nw_p90[i] = np.percentile(nw, [10, 50, 90])

        if trace:
            g = lambda a: float(a[0])
            trace_rows.append({
                "year": int(yr), "age": primary_age,
                "income": g(net_income), "ss": ss_income,
                "spend": g(spend), "contrib": contrib_out,
                "draw_taxable": g(draws.get("taxable", np.zeros(1))),
                "draw_pretax": g(draws.get("pretax", np.zeros(1))),
                "draw_roth": g(draws.get("roth", np.zeros(1))),
                "draw_wl": g(draws.get("wl", np.zeros(1))),
                "draw_rm": g(draws.get("rm", np.zeros(1))),
                "draw_b457": g(draws.get("b457", np.zeros(1))),
                "rmd": g(rmd_net) if isinstance(rmd_net, np.ndarray) else rmd_net,
                "ladder": ladder_conv,
                "cash": g(cash), "taxable": g(taxable), "pretax": g(pretax),
                "roth": g(roth), "b457": g(b457p + b457r), "wl": g(wl),
                "gmult": float(gmult[0]),
                "k529": float(sum(k529[k][0] for k in k529)),
                "home_eq": g(home) - g(rm_drawn) - mortgage_bal,
                "nw": g(nw),
            })

    # mortality-weighted ruin: a ruin only "counts" with the probability
    # that at least one spouse is still alive when it happens (SSA 2019)
    p_alive = either_alive_curve(
        [(this_year - pp["birth_year"], pp.get("sex", "female"))
         for pp in people],
        n_years,
    )
    ruin_weight = np.zeros(n_sims)
    ruined_mask = ~alive
    if ruined_mask.any():
        idx = ruin_year[ruined_mask] - years[0]
        ruin_weight[ruined_mask] = p_alive[idx]
    health_mortality = (1.0 - ruin_weight.mean()) * 100

    estate = (
        cash
        + taxable * (1 - taxable_tax_rate)
        + pretax * (1 - pretax_eff)
        + roth
        + b457p * (1 - pretax_eff) + b457r
        + wl
        + home * (1 - home_cfg["selling_costs"])
        - rm_drawn
        - mortgage_bal
    )
    # MA estate tax on the terminal estate (exclusion 2M, or 4M with
    # credit-shelter trusts — see [taxes] ma_estate_exclusion). The
    # threshold is nominal and unindexed; using today's-real dollars
    # here is generous to the estate.
    ma_estate_paid = np.zeros(n_sims)
    if tx.get("ma_estate_tax", False):
        from finsim.tax import ma_estate_tax
        ma_estate_paid = ma_estate_tax(
            np.maximum(estate, 0.0),
            tx.get("ma_estate_exclusion", 2_000_000))
        estate = estate - ma_estate_paid

    return {
        "years": years,
        "alive": alive,
        "ruin_year": ruin_year,
        "estate": estate,
        "college_funded": college_funded,
        "college_need": college_need_per_kid,
        "college_at_start": college_at_start,
        "k529_end": k529,
        "wd_tax_paid": wd_tax_paid,
        "penalty_paid": penalty_paid,
        "nw_bands": (nw_p10, nw_p50, nw_p90),
        "health_mortality": health_mortality,
        "tax_rates": {"pretax": pretax_rate_t, "cg": cg_rate_t,
                      "ladder": ladder_rate_t, "lti": lti_rate_t},
        "ma_estate_paid": ma_estate_paid,
        "trace": trace_rows,
        "strategy": strat_name,
    }


def solve_college_monthly(cfg, buckets, target_pct=1.0):
    """Deterministic required monthly contribution per kid to have the
    remaining UMass cost saved in the 529 when college starts."""
    m = cfg["market"]
    mu = m["equity_weight"] * m["equity_real_return"] + (1 - m["equity_weight"]) * m["bond_real_return"]
    coll = cfg["college"]
    this_year = date.today().year
    out = {}
    for kid in coll["kids"]:
        name, start = kid["name"], kid["start_year"]
        need = sum(
            coll["target_today"] / coll["years_in_college"]
            * (1 + coll["real_cost_growth"]) ** (start + j - this_year)
            for j in range(coll["years_in_college"])
        ) * target_pct

        def balance_at_start(monthly):
            bal = buckets["529"][name]
            for yr in range(this_year, start):
                bal = bal * (1 + mu) + monthly * 12
            return bal

        lo, hi = 0.0, 5000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if balance_at_start(mid) < need:
                lo = mid
            else:
                hi = mid
        out[name] = {"monthly": hi, "need_at_start": need,
                     "at_start_current": balance_at_start(coll["monthly_contrib"])}
    return out


# ── Reporting ───────────────────────────────────────────────────────────

def report(cfg, buckets, res, scenario, as_json):
    alive = res["alive"]
    health = alive.mean() * 100
    ruined = res["ruin_year"][~alive]
    estate = res["estate"]

    out = {
        "scenario": scenario or "base",
        "strategy": res["strategy"],
        "health_pct": round(health, 1),
        "ruin_pct": round(100 - health, 1),
        "ruin_year_p5": int(np.percentile(ruined, 5)) if ruined.size else None,
        "ruin_year_median": int(np.median(ruined)) if ruined.size else None,
        "estate_p5": round(float(np.percentile(estate, 5))),
        "estate_median": round(float(np.median(estate))),
        "wd_tax_median": round(float(np.median(res["wd_tax_paid"]))),
        "penalty_median": round(float(np.median(res["penalty_paid"]))),
        "college": {},
        "buckets_today": {
            k: round(v) for k, v in buckets.items() if not isinstance(v, dict)
        },
    }
    for name, funded in res["college_funded"].items():
        need = res["college_need"].get(name, 0.0)
        pct = funded / need * 100 if need else np.zeros_like(funded)
        out["college"][name] = {
            "pct_funded_p10": round(float(np.percentile(pct, 10)), 1),
            "pct_funded_median": round(float(np.percentile(pct, 50)), 1),
            "pct_funded_p90": round(float(np.percentile(pct, 90)), 1),
            "total_need_real": round(need),
        }

    if as_json:
        print(json.dumps(out, indent=2))
        return

    W = 62
    print("=" * W)
    print(f"  GOALS SIMULATION — scenario: {out['scenario']}  strategy: {out['strategy']}")
    print("=" * W)
    hp = out["health_pct"]
    flag = "✅" if hp >= 95 else ("🟡" if hp >= 85 else "🔴")
    print(f"\n  {flag} HEALTH: {hp:.1f}%  (paths that never run out of money)")
    print(f"     mortality-weighted: {res['health_mortality']:.1f}%  (ruin counted only"
          f" while someone is likely alive, SSA 2019)")
    print(f"     target: ≥95%  |  ruin in {out['ruin_pct']:.1f}% of paths", end="")
    if out["ruin_year_median"]:
        print(f" (median ruin year {out['ruin_year_median']}, 5th pct {out['ruin_year_p5']})")
    else:
        print()
    print(f"\n  Terminal estate (real $): median {out['estate_median']:,}  |  p5 {out['estate_p5']:,}")
    print(f"  Withdrawal taxes+penalties, median: {out['wd_tax_median']:,}"
          f"  (penalties alone: {out['penalty_median']:,})")
    print("\n  College (% of UMass target funded from each 529):")
    for name, c in out["college"].items():
        print(
            f"    {name:6s} p10 {c['pct_funded_p10']:6.1f}%   "
            f"median {c['pct_funded_median']:6.1f}%   p90 {c['pct_funded_p90']:6.1f}%"
        )
    print("\n  Buckets today (from ledger):")
    b = out["buckets_today"]
    for k in ("cash", "taxable", "pretax", "roth", "b457", "hsa", "lti",
              "whole_life", "home_value", "mortgage", "student_loans", "utma"):
        print(f"    {k:14s} {b[k]:>12,}")
    for kid, v in buckets["529"].items():
        print(f"    529 {kid:10s} {round(v):>12,}")
    print("=" * W)


def print_trace(res):
    cols = ("year age | income  spend contrib | drawTx drawPre drawRoth "
            "drawWL drawRM  rmd | taxable  pretax    roth     wl   529 | networth")
    print(cols)
    for t in res["trace"]:
        print(
            f"{t['year']} {t['age']:3d} | {t['income']/1000:6.0f} {t['spend']/1000:6.0f}"
            f" {t['contrib']/1000:7.0f} | {t['draw_taxable']/1000:6.0f}"
            f" {t['draw_pretax']/1000:7.0f} {t['draw_roth']/1000:8.0f}"
            f" {t['draw_wl']/1000:6.0f} {t['draw_rm']/1000:6.0f} {t['rmd']/1000:4.0f} |"
            f" {t['taxable']/1000:7.0f} {t['pretax']/1000:7.0f} {t['roth']/1000:7.0f}"
            f" {t['wl']/1000:6.0f} {t['k529']/1000:5.0f} | {t['nw']/1000:8.0f}"
        )


def main(argv=None, root=None):
    """CLI entry. `root` is where goals.toml / accounts.toml / main.bc /
    balances.json are looked for by default (cwd if None)."""
    root = Path(root or ".")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goals", type=Path, default=root / "goals.toml",
                    help="assumptions + scenarios (default: goals.toml)")
    ap.add_argument("--accounts", type=Path, default=root / "accounts.toml",
                    help="account registry (default: accounts.toml)")
    ap.add_argument("--ledger", type=Path, default=root / "main.bc",
                    help="beancount ledger to read balances from (default: main.bc)")
    ap.add_argument("--scenario", help="named scenario from goals.toml")
    ap.add_argument("--list-scenarios", action="store_true")
    ap.add_argument("--strategy", choices=list(STRATEGIES))
    ap.add_argument("--deterministic", action="store_true",
                    help="single path at expected returns; prints year table")
    ap.add_argument("--sims", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--dump-balances", metavar="PATH", nargs="?",
                    const=str(root / "balances.json"),
                    help="write the ledger bucket snapshot (the layer-2 "
                         "artifact goals.toml pairs with) and exit")
    ap.add_argument("--balances", metavar="PATH",
                    help="read buckets from a snapshot instead of the ledger")
    args = ap.parse_args(argv)

    cfg, scenario_names = load_config(args.scenario, path=args.goals)
    if args.list_scenarios:
        print("\n".join(scenario_names) or "(none)")
        return

    if args.dump_balances:
        buckets = get_buckets(args.ledger, accounts=args.accounts)
        snap = {"as_of": str(date.today()), "source": str(args.ledger), "buckets": buckets}
        Path(args.dump_balances).write_text(json.dumps(snap, indent=1))
        print(f"wrote {args.dump_balances}")
        return

    if args.balances:
        buckets = json.loads(Path(args.balances).read_text())["buckets"]
    else:
        buckets = get_buckets(args.ledger, accounts=args.accounts)
    res = simulate(
        cfg,
        buckets,
        n_sims=args.sims if args.sims is not None else cfg["simulation"]["n_sims"],
        seed=args.seed if args.seed is not None else cfg["simulation"]["seed"],
        strategy=args.strategy,
        deterministic=args.deterministic,
    )
    if args.deterministic:
        if args.json:
            print(json.dumps(res["trace"], indent=2))
        else:
            print_trace(res)
    else:
        report(cfg, buckets, res, args.scenario, args.json)


if __name__ == "__main__":
    main()

"""Progressive tax math: federal brackets, LTCG stacking, MA, gross-up.

The engine currently runs on effective rates (goals.toml [taxes]); this
module is the tested bracket-accurate layer for the work that needs it
(Roth-ladder sizing, gross-up of withdrawals, MFS-vs-MFJ pricing).
The gross-up binary search is ported from the retirement-optimizer
audit (its one correct tax idea), fixed to take a per-call penalty.

CALIBRATION: tables are 2025 (post-OBBBA, rates permanent). Update the
threshold dollars annually; every function takes the tables as data so
tests and callers can pin any year.
"""

# (lower_threshold, rate); taxable income above the last threshold pays
# the last rate. 2025, post-OBBBA.
FEDERAL = {
    "mfj": [(0, 0.10), (23_850, 0.12), (96_950, 0.22), (206_700, 0.24),
            (394_600, 0.32), (501_050, 0.35), (751_600, 0.37)],
    "single": [(0, 0.10), (11_925, 0.12), (48_475, 0.22), (103_350, 0.24),
               (197_300, 0.32), (250_525, 0.35), (626_350, 0.37)],
}
# Calibration year of every table in this module (brackets, standard
# deduction, FPL, IRMAA, thresholds). Bump it when the tables are updated;
# downstream freshness reports read it.
TABLE_YEAR = 2025

STANDARD_DEDUCTION = {"mfj": 31_500, "single": 15_750}  # 2025 OBBBA
# long-term capital gains: 0/15/20 thresholds on TAXABLE income, gains
# stack on top of ordinary income
LTCG = {
    "mfj": [(0, 0.0), (96_700, 0.15), (600_050, 0.20)],
    "single": [(0, 0.0), (48_350, 0.15), (533_400, 0.20)],
}
# ── State tax registry — add a state by adding an entry ────────────────
# rate: flat income tax (progressive states can extend with brackets);
# taxes_ss: whether the state taxes Social Security benefits;
# estate_*: state estate tax (credit-table style; [] = no estate tax).
MA_RATE = 0.05                   # kept for MA convenience/back-compat
MA_SURTAX_RATE = 0.04            # "millionaire tax"
MA_SURTAX_THRESHOLD = 1_083_150  # 2025, inflation-indexed


def tax_from_brackets(taxable, brackets):
    """Walk a [(lower, rate), ...] table."""
    taxable = max(0.0, taxable)
    tax = 0.0
    for i, (lo, rate) in enumerate(brackets):
        hi = brackets[i + 1][0] if i + 1 < len(brackets) else float("inf")
        if taxable <= lo:
            break
        tax += (min(taxable, hi) - lo) * rate
    return tax


def marginal_rate(taxable, brackets):
    rate = brackets[0][1]
    for lo, r in brackets:
        if taxable >= lo:
            rate = r
    return rate


def federal_ordinary(gross, filing="mfj", deduction=None):
    """Federal tax on ordinary income after the standard deduction."""
    ded = STANDARD_DEDUCTION[filing] if deduction is None else deduction
    return tax_from_brackets(gross - ded, FEDERAL[filing])


def ltcg_tax(gain, ordinary_taxable, filing="mfj"):
    """Federal LTCG: the gain stacks on top of ordinary taxable income
    and each slice pays the 0/15/20 rate of the band it lands in."""
    ordinary_taxable = max(0.0, ordinary_taxable)
    top = ordinary_taxable + max(0.0, gain)
    return (tax_from_brackets(top, LTCG[filing])
            - tax_from_brackets(ordinary_taxable, LTCG[filing]))


def state_tax(taxable, state="MA"):
    """State income tax on ordinary income."""
    cfg = STATES[state]
    taxable = max(0.0, taxable)
    return (taxable * cfg["rate"]
            + max(0.0, taxable - cfg["surtax_threshold"]) * cfg["surtax_rate"])


def ma_tax(taxable):
    """MA flat 5% plus the 4% surtax on income over ~$1.08M."""
    return state_tax(taxable, "MA")


def total_ordinary(gross, filing="mfj", state="MA"):
    """Federal + state on ordinary income (no separate state deduction
    modeled; both on the same gross)."""
    return federal_ordinary(gross, filing) + state_tax(gross, state)


def ss_taxable_fraction(other_agi, ss_total, filing="mfj"):
    """Federal taxable share of Social Security via the provisional-
    income worksheet (thresholds 32k/44k MFJ, unindexed). Returns 0-0.85."""
    if ss_total < 1e-6:
        return 0.0
    t1, t2 = (32_000, 44_000) if filing == "mfj" else (25_000, 34_000)
    prov = other_agi + 0.5 * ss_total
    if prov <= t1:
        taxable = 0.0
    elif prov <= t2:
        taxable = min(0.5 * (prov - t1), 0.5 * ss_total)
    else:
        taxable = min(0.85 * ss_total,
                      0.85 * (prov - t2) + min(0.5 * (t2 - t1), 0.5 * ss_total))
    return min(0.85, taxable / ss_total)


def taxable_ss_of(w, ss_total, filing="mfj", deflator=1.0):
    """Federal-taxable SS given non-SS ordinary income w. `deflator`
    scales the (nominal, frozen-since-1984) 32k/44k thresholds into
    real dollars — the 'tax torpedo' worsens over a real-dollar
    horizon."""
    if ss_total < 1e-6:  # incl. denormal-float noise
        return 0.0
    t1, t2 = (32_000, 44_000) if filing == "mfj" else (25_000, 34_000)
    t1, t2 = t1 * deflator, t2 * deflator
    prov = w + 0.5 * ss_total
    if prov <= t1:
        return 0.0
    if prov <= t2:
        return min(0.5 * (prov - t1), 0.5 * ss_total)
    return min(0.85 * ss_total,
               0.85 * (prov - t2) + min(0.5 * (t2 - t1), 0.5 * ss_total))


def ordinary_tax_grid(filing="mfj", ss_total=0.0, ss_deflator=1.0,
                      x_max=3_000_000, state="MA"):
    """Piecewise-linear composite tax T(w) of NON-SS ordinary income w:
    federal on (w + taxable_ss(w) - deduction) + state on w (or on
    w + taxable SS where the state taxes benefits). Exact between grid
    points: breakpoints include the SS phase-in kinks and every federal
    edge inverted through the phase-in, so the 1.5x/1.85x 'tax torpedo'
    slopes are represented exactly. Returns (x_grid, tax_grid)."""
    import numpy as np
    ded = STANDARD_DEDUCTION[filing]
    st = STATES[state]

    def T(w):
        tss = taxable_ss_of(w, ss_total, filing, ss_deflator)
        fed_taxable = w + tss - ded
        state_base = w + (tss if st["taxes_ss"] else 0.0)
        return (tax_from_brackets(fed_taxable, FEDERAL[filing])
                + state_tax(state_base, state))

    xs = {0.0, x_max, st["surtax_threshold"]}
    # SS phase-in kinks in w-space
    if ss_total >= 1e-6:
        t1, t2 = (32_000, 44_000) if filing == "mfj" else (25_000, 34_000)
        t1, t2 = t1 * ss_deflator, t2 * ss_deflator
        w1 = max(0.0, t1 - 0.5 * ss_total)
        w2 = max(w1, t2 - 0.5 * ss_total)
        cap_base = min(0.5 * (t2 - t1), 0.5 * ss_total)
        # cap: 0.85*(w + 0.5*ss - t2) + cap_base == 0.85*ss
        w3 = max(w2, 0.5 * ss_total + t2 - cap_base / 0.85)
        # small-SS plateau: the 50% ramp caps at 0.5*ss BEFORE t2 when
        # ss < t2 - t1, flattening between w4 and w2 (found by the
        # Hypothesis exactness property)
        w4 = min(w2, t1 + 0.5 * ss_total)
        xs |= {w1, w2, w3, w4}
        segments = [(0.0, w1, 0.0), (w1, w4, 0.5), (w4, w2, 0.0),
                    (w2, w3, 0.85), (w3, x_max, 0.0)]
    else:
        segments = [(0.0, x_max, 0.0)]
    # federal bracket edges inverted through w + tss(w) = edge + ded
    for lo, _ in FEDERAL[filing]:
        target = lo + ded
        for a, b, slope in segments:
            tss_a = taxable_ss_of(a, ss_total, filing, ss_deflator)
            # on the segment: w + tss_a + slope*(w - a) = target
            w = (target - tss_a + slope * a) / (1 + slope)
            if a - 1e-9 <= w <= b + 1e-9:
                xs.add(min(max(w, 0.0), x_max))
    x = np.array(sorted(v for v in xs if 0 <= v <= x_max))
    return x, np.array([T(v) for v in x])


STATES = {
    "MA": {"rate": MA_RATE, "surtax_rate": MA_SURTAX_RATE,
           "surtax_threshold": MA_SURTAX_THRESHOLD, "taxes_ss": False},
    "none": {"rate": 0.0, "surtax_rate": 0.0,
             "surtax_threshold": float("inf"), "taxes_ss": False},
}


# ── ACA premium tax credit (post-2025: enhanced subsidies expired, the
# 400% FPL cliff is back). expected-contribution % of MAGI by FPL ratio,
# linearly interpolated; above 4.0 the credit is zero.
_ACA_PCT = [(1.00, 0.0207), (1.33, 0.0310), (1.50, 0.0414),
            (2.00, 0.0652), (2.50, 0.0833), (3.00, 0.0983), (4.00, 0.0986)]
FPL_2 = 21_150   # 2025 federal poverty line, 2-person household


def aca_premium_cost(magi, full_premium, fpl=FPL_2):
    """What the household pays toward a benchmark `full_premium` given
    MAGI: expected contribution below 400% FPL, full premium above the
    cliff. Vectorized over magi."""
    import numpy as np
    magi = np.asarray(magi, dtype=float)
    ratio = np.maximum(magi, 1.0) / fpl
    pts = np.array([p for p, _ in _ACA_PCT])
    pcts = np.array([c for _, c in _ACA_PCT])
    pct = np.interp(ratio, pts, pcts)
    contribution = np.minimum(pct * magi, full_premium)
    out = np.where(ratio > 4.0, full_premium, contribution)
    return out if out.shape else float(out)


# IRMAA: annual Medicare B+D surcharge for a COUPLE by MFJ MAGI (2025
# tiers, 2-year lookback ignored). Approximate; unindexed here.
_IRMAA = [(212_000, 1_900), (266_000, 4_700), (334_000, 7_600),
          (400_000, 10_400), (750_000, 11_400)]


def irmaa_couple(magi):
    """Annual IRMAA surcharge for a couple at the given MAGI."""
    import numpy as np
    magi = np.asarray(magi, dtype=float)
    out = np.zeros_like(magi)
    for lo, add in _IRMAA:
        out = np.where(magi > lo, float(add), out)
    return out if out.shape else float(out)


# MA estate tax (post-2023 reform): compute the old federal
# state-death-tax-credit table on the whole estate, minus the credit on
# the $2M exclusion (~$99,600). Table keys: (adjusted-estate lower
# bound, rate); "adjusted" = estate - 60,000 per the old federal form.
# APPROXIMATE — verify with the attorney; thresholds are NOT
# inflation-indexed (so using today's-real dollars is generous).
_MA_CREDIT_TABLE = [
    (0, 0.0), (40_000, 0.008), (90_000, 0.016), (140_000, 0.024),
    (240_000, 0.032), (440_000, 0.04), (640_000, 0.048), (840_000, 0.056),
    (1_040_000, 0.064), (1_540_000, 0.072), (2_040_000, 0.08),
    (2_540_000, 0.088), (3_040_000, 0.096), (3_540_000, 0.104),
    (4_040_000, 0.112), (5_040_000, 0.12), (6_040_000, 0.128),
    (7_040_000, 0.136), (8_040_000, 0.144), (9_040_000, 0.152),
    (10_040_000, 0.16),
]
MA_ESTATE_EXCLUSION = 2_000_000


def _ma_credit(estate):
    return tax_from_brackets(max(0.0, estate - 60_000), _MA_CREDIT_TABLE)


def ma_estate_tax(estate, exclusion=MA_ESTATE_EXCLUSION):
    """MA estate tax due on a taxable estate. Zero at or under the
    exclusion; above it, the credit-table tax less the credit on the
    exclusion amount. Accepts numpy arrays."""
    import numpy as np
    estate = np.asarray(estate, dtype=float)
    credit_at_excl = _ma_credit(exclusion)
    gross = np.zeros_like(estate)
    adj = np.maximum(0.0, estate - 60_000)
    for i, (lo, rate) in enumerate(_MA_CREDIT_TABLE):
        hi = (_MA_CREDIT_TABLE[i + 1][0]
              if i + 1 < len(_MA_CREDIT_TABLE) else np.inf)
        gross += np.clip(adj - lo, 0.0, hi - lo) * rate
    out = np.where(estate > exclusion,
                   np.maximum(0.0, gross - credit_at_excl), 0.0)
    return out if out.shape else float(out)


def eff_rate_on(amount, other_income=0.0, filing="mfj", state="MA"):
    """Effective fed+state rate on `amount` of ordinary income stacked
    on top of other_income."""
    if amount <= 0:
        return 0.0
    return (total_ordinary(other_income + amount, filing, state)
            - total_ordinary(other_income, filing, state)) / amount


def cg_eff_rate_on(gain, other_taxable=0.0, filing="mfj", state="MA"):
    """Effective fed LTCG rate (stacked) + state rate on `gain`."""
    if gain <= 0:
        return STATES[state]["rate"]
    return ltcg_tax(gain, other_taxable, filing) / gain + STATES[state]["rate"]


def gross_up(net_needed, other_income=0.0, filing="mfj", penalty=0.0,
             tol=0.5, iters=60, state="MA"):
    """How much must be withdrawn (as ordinary income on top of
    other_income) to net `net_needed` after federal + MA + penalty?
    Binary search; the marginal rate moves as the withdrawal grows."""
    if net_needed <= 0:
        return 0.0

    def net_of(gross_wd):
        tax = (total_ordinary(other_income + gross_wd, filing, state)
               - total_ordinary(other_income, filing, state))
        return gross_wd - tax - gross_wd * penalty

    lo, hi = net_needed, net_needed * 4 + 10_000
    for _ in range(iters):
        mid = (lo + hi) / 2
        if net_of(mid) < net_needed:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi

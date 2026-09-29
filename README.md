# finsim

**The behavioral, auditable, path-wise Monte Carlo engine for household
financial planning** — the implementable-policy complement to
perfect-foresight optimizers.

finsim simulates *policies*: rules a real household can actually follow,
evaluated year by year against thousands of return sequences, with the
tax code computed exactly. It answers questions like *"does this
spending rule survive a 1970s sequence?"* and *"what does a one-time
$10k purchase cost my terminal estate?"* — questions an optimizer with
foresight of the return path structurally cannot.

## What it models

- **Sequence-of-returns risk, honestly.** Historical bootstrap of
  1953–2019 real returns with per-asset autocorrelation (RPM-style),
  recentered to your expected returns. i.i.d. normal available for
  comparison; averaged paths are a non-goal.
- **Accounts as first-class rules.** Pre-tax / Roth (basis + 5-year
  conversion seasoning as annual vintages) / HSA / governmental 457(b)
  with its separation-from-service rule / 529 per child / whole-life
  cash value / reverse mortgage (age- and LTV-gated). Contribution
  routing with limits, RMDs on the Uniform Lifetime Table.
- **Exact progressive taxes, per simulation path.** Piecewise-linear
  federal + state tax grids per year; every ordinary-income event
  (RMDs, conversions, withdrawals) stacks marginally on a per-path
  running income; gross-from-net inverts the same piecewise function.
  Social Security taxation via the provisional-income worksheet with
  its **unindexed thresholds deflating in real terms** (the tax
  torpedo, exactly). LTCG stacking. ACA premium subsidies with the
  400% FPL cliff, **fed by MAGI including planned Roth conversions**.
  IRMAA on MAGI(t−2). NIIT. State estate tax on terminal estates.
- **Behavior.** Canonical Guyton-Klinger guardrails (capital
  preservation, prosperity, and inflation rules — all knobs tunable)
  plus a simpler threshold style. Success can be **mortality-weighted**:
  a ruin only counts with the SSA joint probability that someone is
  alive to experience it.
- **Goals, not just retirement dates.** College-by-18 alongside
  lifetime solvency, and one-line sensitivity of the whole plan to a
  single purchase — the long-term price of *save vs. spend*.

## Design doctrine

Two ideas carry the architecture (see `docs/ARCHITECTURE.md`):

1. **Policies, not plans.** No rule may read future state — enforced
   structurally and tested (inject a poisoned future; outputs through
   year *k* must be identical).
2. **State exists only if a rule reads it.** The tax code is the
   schema. The engine doesn't care about tax lots in your 401(k)
   because *no rule reads them* — everything exits as ordinary income.
   Roth conversions are annual vintages because the 5-year clock reads
   years, not dates. The DSL stays minimal because the rules define
   the state vector, not modeling ambition.

## Quickstart

```bash
uv sync
uv run pytest -q            # the audit trail: ~50 tests in ~2s
uv run python examples/run.py
```

Or drive it from the command line, from the directory that holds your
`goals.toml` (and `accounts.toml` + a beancount ledger if you use one),
or with explicit paths:

```bash
finsim                                   # base case Monte Carlo
finsim --scenario downshift              # a named scenario
finsim --deterministic                   # expected-return year table
finsim --balances balances.json          # run from a snapshot, no ledger
finsim --goals g.toml --accounts a.toml --ledger main.bc --dump-balances
```

`finsim.testing` exports the synthetic-household builders the unit
tests use (`mini_cfg`, `buckets`, `det`), so a private repo can write
its own invariant tests without copying fixtures.

Inputs are three declarative files (see `examples/`):

| File | What it holds |
|---|---|
| `goals.toml` | People, income phases, spending, goals, market model, tax settings, named scenarios (deep-merged overrides) |
| `balances.json` | A dated snapshot of balances by tax bucket — from any source of truth you like |
| `accounts.toml` | Optional registry mapping a [beancount](https://beancount.github.io/) ledger into buckets, with an audit that proves every funded account is claimed |

```python
from finsim import load_config, simulate

cfg, scenarios = load_config(None, path="examples/goals.toml")
res = simulate(cfg, buckets, n_sims=5000, seed=42)
res["alive"].mean()          # success rate
res["health_mortality"]      # mortality-weighted success
res["tax_rates"]             # the per-year rate curves it derived
```

## Testing story

Five layers of defense, because a planning engine you can't audit is a
vibes generator:

1. **Closed-form unit tests** — zero-return deterministic runs make
   every account rule exactly checkable.
2. **Property-based (Hypothesis)** — laws over the whole input space:
   gross-up round-trips, grid-vs-brute tax exactness, monotonicity,
   conservation. (These have caught real bugs spot-checks missed.)
3. **Invariants** — less spending / more assets / flexibility can never
   hurt; identical seeds reproduce identically; no future peeking.
4. **Golden regressions** — a pinned deterministic trajectory; any
   change to the baseline must be a conscious commit.
5. **Cross-checks** — the gross-up solver and the grid inversion are
   independent implementations that must agree to $2.

## Calibration honesty

Tax tables are 2025 (post-OBBBA) and marked for annual recalibration.
Documented approximations: capital-gains stacking uses a per-year
effective rate (ordinary income is exact); the SS taxable share uses a
reference withdrawal; ACA/IRMAA/estate thresholds are held in real
dollars. Every approximation is written down next to its code.

## Landscape

How this engine relates to Owl, monteplan, TPAW, FI Calc and friends:
`docs/COMPARISON.md`.

## License

Apache-2.0. Nothing here is investment, tax, or legal advice — it is a
calculator with a test suite.

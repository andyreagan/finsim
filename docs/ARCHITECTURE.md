# Architecture

## Doctrine 1: Policies, not plans

An optimizer (LP/MILP, e.g. Owl) solves for the optimal **plan** given
one return path — so each solution has **perfect foresight** of that
path. Averaged over Monte Carlo draws, it yields an upper bound on
achievable outcomes, not a followable strategy.

finsim simulates **policies**: pure functions of **observables at time
t** — no foresight, ever. It answers the question an LP structurally
can't: *does an implementable rule hold up across sequences?*

**Hard rule:** no policy function may read future state. This is
enforced structurally (the year loop only carries running state
forward) and tested: `simulate(..., returns=...)` accepts injected
return matrices, and the poisoned-future invariant runs two matrices
that agree through year *k* and diverge violently after — every output
through year *k* must be bit-identical.

## Doctrine 2: State exists only if a rule reads it

The tax code is the schema. Enumerate the rules, take the union of
variables their formulas reference — that union is the state vector.
Nothing else earns a slot.

Settled applications of the principle:

| Domain | Keep | Drop | Why |
|---|---|---|---|
| Pre-tax (401k/IRA/govt-457b) | Balance, owner age | Tax lots | Everything exits as ordinary income; no rule reads lots. |
| Roth | Conversion **vintages as annual buckets** (5-yr clocks start with the tax year), contribution basis | Lots, exact dates | Ordering + seasoning read year granularity only. |
| Taxable | Aggregate basis fraction | Per-lot detail | LTCG stacking reads the basis fraction; only specific-ID harvesting would read lots — out of scope for core. |
| IRMAA | MAGI(t−2) lag buffer | — | The rule dictates the state. |
| Account types | Flag tuple: (tax-in, tax-growth, tax-out, penalty schedule, limits, forced flows) | Per-type class hierarchies | Govt 457(b) = pre-tax + no-penalty-after-separation. HSA = triple-exempt + post-65 pre-tax fallback. No new machinery per type. |
| Timestep | Annual, in tax years | Monthly | Every rule that matters resolves annually. |

*(The current engine implements the flag-tuple idea as explicit bucket
branches; collapsing them into declared tuples is the intended
direction as account types grow.)*

## Derived observables are first-class

Every smart policy is "distance to a threshold," so the engine derives
per-year, per-path:

- headroom to the next federal/state bracket edge (the tax grids)
- ACA subsidy position (MAGI including planned conversions vs the
  400% FPL cliff)
- IRMAA tier exposure against MAGI(t−2)
- SS torpedo phase-in position (provisional income vs the *unindexed*
  thresholds, deflating in real terms)
- guardrail ratio (current withdrawal rate ÷ initial)
- joint survival probability (SSA tables)

With these, guardrails, bracket-fill conversions, and cliff avoidance
are each one-liners against shared primitives.

## Guyton-Klinger guardrails (implementation spec)

From Guyton & Klinger (2006); all four parameters tunable
(`spending.guardrails`, `style = "gk"`):

- **Capital preservation:** current WR > `upper` (1.20) × initial WR →
  cut real spending by `adjust` (10%), floored at `floor`.
- **Prosperity:** current WR < `lower` (0.80) × initial WR → raise by
  `adjust`, capped at `ceiling`.
- **Inflation rule:** after a negative portfolio-return year, skip the
  inflation adjustment (in this real-dollar engine: an unrecovered
  real cut of ~CPI).

All three rules read only the guardrail ratio and last year's return
sign — which is the proof the observable set is right.

## The exact-tax machinery

Each year builds a piecewise-linear composite tax function T(w) of
non-SS ordinary income (federal after deduction on w + taxable-SS(w),
plus state), with breakpoints at every federal edge *inverted through
the SS phase-in* and the phase-in's own kinks (including the small-SS
plateau — found by a Hypothesis property, not by inspection). Every
ordinary-income event taxes its slice via `np.interp`, stacking on a
per-path running income level; gross-from-net inverts
H(t) = t(1−penalty) − T(t) on the same grid. Exact, vectorized, and
cross-checked against an independent binary-search solver.

## Layer boundary

The engine consumes only a declarative contract: `goals.toml`
(assumptions + scenarios), `balances.json` (a dated bucket snapshot),
optionally `accounts.toml` (a registry mapping a beancount ledger into
buckets, with an audit proving every funded account is claimed). No
account numbers, no institutions, no PII beyond birth years and a
state — the DSL boundary is also a privacy boundary.

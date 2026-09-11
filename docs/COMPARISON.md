# Landscape: community retirement engines

Where finsim sits among open-source and community-supported retirement
simulation tools, scored against the capability checklist below.
Preliminary scores from a 2026-09 survey; corrections welcome by issue
or PR.

## The checklist

1. **Sequence risk** — historical bootstrap with autocorrelation (iid
   normal and averaged paths are non-goals)
2. **Accounts as first-class** — pre-tax/Roth/HSA/govt-457(b)
   separation rule/529/whole-life/reverse mortgage; contribution
   limits; Roth basis + 5-yr seasoning; RMDs
3. **Exact progressive tax math** — fed + state brackets per year; SS
   torpedo with unindexed-threshold deflation; LTCG stacking; ACA
   cliff fed by MAGI incl. conversions; IRMAA; NIIT; state estate tax
4. **Behavior** — spending guardrails; mortality-weighted success;
   goals-based sensitivity
5. **Engineering** — declarative DSL separate from the engine;
   auditable account registry; property-based + golden + invariant
   tests

## The field

| Project | Lang / License | Community health | Checklist fit | Notes |
|---|---|---|---|---|
| [Owl](https://github.com/mdlacasse/Owl) (`owlplanner`) | Python / GPL-3.0 | Very active; effectively 1 core maintainer | ~50–60% | MILP withdrawal/Roth-conversion **optimizer**. Deepest tax stack in OSS: fed + 50 states, LTCG, NIIT, IRMAA, ACA, RMDs, Roth maturation. Weak on bootstrap sequence risk, behavior rules, and DSL/testing. |
| [monteplan](https://github.com/engineerinvestor/monteplan) | Python | New, tiny | ~35% | Vectorized MC; block bootstrap, Student-t, regimes; Guyton-Klinger, VPW; fed brackets + LTCG + RMDs. Closest in *paradigm*; shallow tax model. |
| [TPAW Planner](https://github.com/bengmathew/tpaw) | TypeScript / open | Solo dev, active; strong Bogleheads thread | ~20% | Amortization-based withdrawal, total-portfolio framing. Different paradigm. |
| [FI Calc](https://ficalc.app) | JS / open | Solo dev | ~15% | Historical cycles; rich withdrawal-strategy menu; no tax layer. |
| cFIREsim-open | JS | Dormant | ~15% | Historical cycles only. |
| Bogleheads RPM / VPW | Spreadsheets | Forum-supported | ~15% | Deterministic / table-driven; no license; not engines. (finsim's historical return generator follows RPM's methodology.) |
| Rich, Broke or Dead (engaging-data) | Closed web | n/a | n/a | Signature mortality-weighted success — validates that idea; not forkable. |
| FIRECalc | Closed | Stale | n/a | Historical cycles. |

Commercial closed tools (ProjectionLab, Boldin, Pralana, MaxiFi)
validate demand for deep tax modeling; none are auditable or
scriptable.

## Positioning

Nothing community-supported reaches ~70% of the checklist. Owl is the
serious contender at roughly half — but Owl is an **optimizer**: a
linear program solves for the best *plan* given a return path, which
gives each solution perfect foresight of that path. Averaged over
Monte Carlo draws that is an upper bound on what's achievable, not a
strategy a household can follow.

finsim is the complement: it simulates **implementable policies** —
pure functions of what is observable at time *t* — and asks whether
they hold up across sequences. The two paradigms answer different
questions and pair well: use an optimizer to find the frontier, use
finsim to test whether a rule you could actually follow gets close
to it.

Capabilities that exist in finsim and (as of this survey) nowhere else
in open source: autocorrelated historical bootstrap with recentering,
the ACA cliff fed by conversion-inclusive MAGI, the SS torpedo with
deflating unindexed thresholds, SSA joint-survival-weighted success,
one-line goal sensitivity to a single purchase, and a declarative
scenario DSL backed by property-based testing.

## A note on Owl and licensing

Owl is GPL-3.0; finsim is Apache-2.0. We treat Owl's published tax
constants and results as a *specification* to cross-check against
(independent implementation), never as code to copy. Where paradigms
overlap — the pure tax layer — matching Owl's numbers is a test goal.

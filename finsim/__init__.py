"""finsim — the household simulation engine (layer 3).

Consumes the layer-2 contract (goals.toml + accounts.toml +
balances.json / the beancount ledger) and produces Monte Carlo
projections, scenario comparisons, and reports. Audited by tests/.
"""

from finsim.engine import (  # noqa: F401
    BUCKET_SCALARS,
    ROOT,
    STRATEGIES,
    ULT,
    gen_returns,
    get_buckets,
    load_config,
    load_registry,
    simulate,
    solve_college_monthly,
)

"""finsim — the household simulation engine (layer 3).

Consumes the layer-2 contract (goals.toml + accounts.toml +
balances.json, or a beancount ledger through the registry) and
produces Monte Carlo projections and scenario comparisons. Nothing
household-specific lives here: people, accounts, amounts and tax
jurisdiction all come from those files.
"""

from finsim.engine import (  # noqa: F401
    BUCKET_SCALARS,
    STRATEGIES,
    ULT,
    gen_returns,
    get_buckets,
    load_config,
    load_registry,
    simulate,
    solve_college_monthly,
)

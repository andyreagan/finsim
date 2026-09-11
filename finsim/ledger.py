"""Beancount ledger loading and USD valuation.

Canonical home for the ledger glue the engine needs (scripts/reports.py
keeps its own copies for presentation; consolidate there eventually).
"""

from pathlib import Path


def load_ledger(main_bc: Path):
    from beancount import loader
    from beancount.core.data import Price

    entries, errors, options = loader.load_file(str(main_bc))

    price_map = {}
    for entry in entries:
        if isinstance(entry, Price):
            c = entry.currency
            if c not in price_map or entry.date > price_map[c][0]:
                price_map[c] = (entry.date, float(entry.amount.number))

    return entries, price_map


def to_usd(balance, price_map):
    total = 0.0
    for pos in balance:
        c = pos.units.currency
        amt = float(pos.units.number)
        if c == "USD":
            total += amt
        elif c in price_map:
            total += amt * price_map[c][1]
    return total

"""Mortality curves and joint spousal survival.

SSA 2019 Period Life Table (deaths per 1,000 by age and sex), ported
from ~/websites/retirement-optimization/backend/api/mortality.py.
Used to mortality-weight the Monte Carlo: a path that runs out of money
after both spouses have likely died is not a lived failure.
"""

import numpy as np

# deaths per 1,000, SSA 2019 Period Life Table, ages 50-120
QX = {
    "male": {
        50: 4.8, 51: 5.2, 52: 5.7, 53: 6.2, 54: 6.8, 55: 7.4, 56: 8.1,
        57: 8.9, 58: 9.7, 59: 10.6, 60: 11.6, 61: 12.7, 62: 13.9,
        63: 15.2, 64: 16.6, 65: 18.2, 66: 19.9, 67: 21.8, 68: 23.9,
        69: 26.2, 70: 28.8, 71: 31.6, 72: 34.8, 73: 38.3, 74: 42.2,
        75: 46.6, 76: 51.5, 77: 57.0, 78: 63.1, 79: 69.9, 80: 77.5,
        81: 86.0, 82: 95.5, 83: 106.2, 84: 118.2, 85: 131.7, 86: 147.0,
        87: 164.3, 88: 183.9, 89: 206.1, 90: 231.4, 91: 260.2, 92: 292.9,
        93: 330.1, 94: 372.4, 95: 420.4, 96: 475.0, 97: 537.1, 98: 607.7,
        99: 688.0, 100: 779.2, 101: 882.6, 102: 999.6, 103: 1131.8,
        104: 1281.1, 105: 1449.6, 106: 1639.7, 107: 1853.9, 108: 2094.9,
        109: 2365.5, 110: 2669.5, 111: 3010.0, 112: 3390.2, 113: 3814.4,
        114: 4287.0, 115: 4812.6, 116: 5396.0, 117: 6042.3, 118: 6756.9,
        119: 7546.4, 120: 8417.4,
    },
    "female": {
        50: 2.9, 51: 3.2, 52: 3.5, 53: 3.8, 54: 4.2, 55: 4.6, 56: 5.0,
        57: 5.5, 58: 6.0, 59: 6.6, 60: 7.2, 61: 7.9, 62: 8.6, 63: 9.4,
        64: 10.3, 65: 11.3, 66: 12.4, 67: 13.6, 68: 14.9, 69: 16.4,
        70: 18.0, 71: 19.8, 72: 21.8, 73: 24.0, 74: 26.4, 75: 29.1,
        76: 32.1, 77: 35.5, 78: 39.3, 79: 43.6, 80: 48.4, 81: 53.8,
        82: 59.9, 83: 66.7, 84: 74.4, 85: 83.1, 86: 92.9, 87: 103.9,
        88: 116.3, 89: 130.4, 90: 146.3, 91: 164.3, 92: 184.6, 93: 207.5,
        94: 233.4, 95: 262.6, 96: 295.5, 97: 332.7, 98: 374.7, 99: 422.2,
        100: 475.8, 101: 536.3, 102: 604.6, 103: 681.5, 104: 767.9,
        105: 864.8, 106: 973.3, 107: 1094.6, 108: 1230.0, 109: 1380.9,
        110: 1548.8, 111: 1735.4, 112: 1942.5, 113: 2171.9, 114: 2425.5,
        115: 2705.2, 116: 3013.1, 117: 3351.5, 118: 3722.8, 119: 4129.5,
        120: 4574.2,
    },
}


def qx(age, sex):
    """Annual death probability at the given age."""
    if age < 50:
        return (1.0 if sex == "male" else 0.7) / 1000.0
    if age > 120:
        return 1.0
    return QX[sex][age] / 1000.0


def survival_curve(current_age, sex, n_years):
    """P(alive at the START of each of the next n_years), from today.

    Element 0 is 1.0 (alive now); element t is the cumulative product of
    surviving years 0..t-1.
    """
    s = np.empty(n_years)
    alive = 1.0
    for t in range(n_years):
        s[t] = alive
        alive *= 1.0 - qx(current_age + t, sex)
    return s


def either_alive_curve(ages_sexes, n_years):
    """P(at least one of the people is alive) per year, independence
    assumed. ages_sexes: list of (current_age, sex)."""
    p_all_dead = np.ones(n_years)
    for age, sex in ages_sexes:
        s = survival_curve(age, sex, n_years)
        p_all_dead *= 1.0 - s
    return 1.0 - p_all_dead


def life_expectancy(current_age, sex, horizon=120):
    """Expected remaining years: sum of cumulative survival probs."""
    n = horizon - current_age + 1
    return float(survival_curve(current_age, sex, n).sum()) - 1.0

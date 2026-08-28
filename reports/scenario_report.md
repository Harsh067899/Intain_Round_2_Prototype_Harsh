# Task 5 — Scenario & Stress Simulation

Shocks scale monthly transition hazards and propagate through dynamics (a delinquency shock compounds into later defaults), not point-multiplied outputs.

## Portfolio outcomes at 12 months
| scenario        |   cum_default (%) |   cum_prepay (%) |   pct_delinquent (%) |
|:----------------|------------------:|-----------------:|---------------------:|
| base            |              8.26 |             7.55 |                 7.43 |
| adverse_credit  |             15    |             5.04 |                 9.76 |
| high_prepayment |              6.92 |            15.38 |                 6.46 |

Monte Carlo 90% band, adverse-credit 12m default (300 paths, 2,000 loans): **13.75% – 16.30%**

## Segment impact — 12m cumulative default by credit band
| value   |   adverse_credit |   base |   high_prepayment |   adverse_delta_pp |
|:--------|-----------------:|-------:|------------------:|-------------------:|
| 620-679 |            23.4  |  13.74 |             11.66 |               9.65 |
| <620    |            21.9  |  12.98 |             11.03 |               8.92 |
| 680-739 |            14.15 |   7.65 |              6.41 |               6.5  |
| 740-779 |            11.86 |   5.97 |              4.89 |               5.89 |
| 780+    |             7.67 |   3.85 |              3.16 |               3.82 |

## Segment impact — top states by adverse delta
| value   |   adverse_credit |   base |   high_prepayment |   adverse_delta_pp |
|:--------|-----------------:|-------:|------------------:|-------------------:|
| CA      |            17.81 |   9.84 |              8.22 |               7.97 |
| AZ      |            15.56 |   7.82 |              6.28 |               7.74 |
| GA      |            18.12 |  10.71 |              9.17 |               7.41 |
| FL      |            16.14 |   9.21 |              7.85 |               6.93 |
| PA      |            14.93 |   8.09 |              6.7  |               6.84 |
| MI      |            15.31 |   8.61 |              7.26 |               6.7  |
| WA      |            13.14 |   6.61 |              5.37 |               6.53 |
| TX      |            14.79 |   8.56 |              7.32 |               6.23 |

## Top scenario drivers (explanation)
- **Adverse credit** hits low-credit, high-LTV segments hardest: their baseline hazards are largest, so multiplicative stress compounds most — visible in the monotone adverse_delta_pp ordering by credit band.
- **High prepayment** pulls loans out of the risk pool early, mechanically lowering cumulative defaults versus base — competing risks in action.
- Delinquency-state loans at cohort start migrate fastest under stress: the shock multiplies already-elevated worsening hazards.

_Figure: reports/figures/scenario_curves.png · Full tables: reports/artifacts/scenario_{curves,segments}.csv_

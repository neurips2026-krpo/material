# semantic spot-check and strict sensitivity

- Audit result: 19/21 valid (90.5%), 2/21 invalid.
- Relevant: 6/7 valid.
- Random: 7/7 valid.
- Conflict: 6/7 valid.
- Both failures are semantic construct failures in `wei_qi_ying_xue`; all mechanical checks passed.

## Audit-informed strict sensitivity

These filters were motivated by the outcome-independent audit and must be labeled post-audit sensitivity analyses.

### relevant (n=83)

| Quality | Base | K-RPO | Δ | 95% CI | Holm p |
|---:|---:|---:|---:|---:|---:|
| 100% | 14.5% | 39.8% | +25.3 pp | — | — |
| 75% | 15.7% | 31.3% | +15.7 pp | [+3.6,+27.7] | 0.02412 |
| 50% | 13.3% | 33.7% | +20.5 pp | [+9.6,+31.3] | 0.001953 |
| 25% | 15.7% | 33.7% | +18.1 pp | [+7.2,+28.9] | 0.00705 |
| 0% | 13.3% | 28.9% | +15.7 pp | [+7.2,+25.3] | 0.00705 |

### random (n=83)

| Quality | Base | K-RPO | Δ | 95% CI | Holm p |
|---:|---:|---:|---:|---:|---:|
| 100% | 14.5% | 39.8% | +25.3 pp | — | — |
| 75% | 19.3% | 41.0% | +21.7 pp | [+9.6,+33.7] | 0.001431 |
| 50% | 16.9% | 41.0% | +24.1 pp | [+13.3,+36.1] | 0.00036 |
| 25% | 15.7% | 41.0% | +25.3 pp | [+14.5,+36.1] | 0.0001477 |
| 0% | 16.9% | 42.2% | +25.3 pp | [+15.7,+34.9] | 2.289e-05 |

### conflict (n=87)

| Quality | Base | K-RPO | Δ | 95% CI | Holm p |
|---:|---:|---:|---:|---:|---:|
| 100% | 20.7% | 34.5% | +13.8 pp | — | — |
| 75% | 18.4% | 32.2% | +13.8 pp | [+3.4,+24.1] | 0.06797 |
| 50% | 9.2% | 24.1% | +14.9 pp | [+6.9,+24.1] | 0.009399 |
| 25% | 12.6% | 21.8% | +9.2 pp | [+0.0,+18.4] | 0.1536 |
| 0% | 4.6% | 10.3% | +5.7 pp | [+0.0,+11.5] | 0.1536 |

## Interpretation

- The stricter relevant-rule analysis strengthens the K-RPO-versus-Base pattern at all four new levels.
- The stricter conflict analysis still shows higher K-RPO point estimates at every level, but only the 50% level survives Holm correction; the original claim that all four conflict contrasts are significant is not robust to this stricter eligibility rule.
- The experiment remains useful as quantitative exploratory evidence, but the rebuttal must disclose the 19/21 spot-check result and avoid claiming perfect semantic perturbation validity.

# knowledge-imperfection results

Status: **PASS**

Primary tables below use the predeclared eligible set.

## relevant

| Quality | n | Base truth | K-RPO truth | Δ (pp) | 95% CI (pp) | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 98 | 19.4% | 37.8% | +18.4 | — | — |
| 75% | 98 | 19.4% | 30.6% | +11.2 | [0.0, 22.4] | 0.1567 |
| 50% | 98 | 15.3% | 31.6% | +16.3 | [6.1, 25.5] | 0.009976 |
| 25% | 98 | 17.3% | 32.7% | +15.3 | [5.1, 25.5] | 0.01777 |
| 0% | 98 | 19.4% | 28.6% | +9.2 | [0.0, 18.4] | 0.1567 |

## random

| Quality | n | Base truth | K-RPO truth | Δ (pp) | 95% CI (pp) | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 98 | 19.4% | 37.8% | +18.4 | — | — |
| 75% | 98 | 21.4% | 37.8% | +16.3 | [6.1, 26.5] | 0.007 |
| 50% | 98 | 19.4% | 39.8% | +20.4 | [10.2, 30.6] | 0.0009747 |
| 25% | 98 | 20.4% | 38.8% | +18.4 | [7.1, 28.6] | 0.004205 |
| 0% | 98 | 20.4% | 39.8% | +19.4 | [10.2, 28.6] | 0.0006261 |

## conflict

| Quality | n | Base truth | K-RPO truth | Δ (pp) | 95% CI (pp) | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 97 | 20.6% | 37.1% | +16.5 | — | — |
| 75% | 97 | 18.6% | 36.1% | +17.5 | [7.2, 27.8] | 0.005493 |
| 50% | 97 | 13.4% | 29.9% | +16.5 | [8.2, 25.8] | 0.003422 |
| 25% | 97 | 15.5% | 28.9% | +13.4 | [4.1, 22.7] | 0.007197 |
| 0% | 97 | 6.2% | 18.6% | +12.4 | [5.2, 19.6] | 0.005493 |

## Amount-matched relevant deletion minus random deletion

| Model | Quality | Δ truth (pp) | 95% CI (pp) | Holm p |
|---|---:|---:|---:|---:|
| base | 75% | -2.0 | [-11.2, 7.1] | 1 |
| base | 50% | -4.1 | [-13.3, 5.1] | 1 |
| base | 25% | -3.1 | [-10.2, 3.1] | 1 |
| base | 0% | -1.0 | [-8.2, 6.1] | 1 |
| krpo | 75% | -7.1 | [-15.3, 1.0] | 0.2869 |
| krpo | 50% | -8.2 | [-15.3, -2.0] | 0.1157 |
| krpo | 25% | -6.1 | [-13.3, 1.0] | 0.2869 |
| krpo | 0% | -11.2 | [-19.4, -3.1] | 0.0509 |

Interpretation boundary: performance under missing knowledge is robustness evidence; following injected false knowledge is not clinical robustness and is reported separately as a safety boundary.

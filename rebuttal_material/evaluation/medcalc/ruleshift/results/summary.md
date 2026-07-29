# Task 10 MedCalc rule-revision results

Paired cases: **n=67**; two prompts per case and model.

| Condition | Control acc. | Revised-rule acc. | Stale-answer rate | Switch success | Revised pair-F1 |
|---|---:|---:|---:|---:|---:|
| base_ruleshift | 55.2% | 56.7% | 11.9% | 47.8% | 74.5% |
| outcome_ruleshift | 49.3% | 47.8% | 11.9% | 44.8% | 75.6% |
| krpo_ruleshift | 56.7% | 56.7% | 7.5% | 52.2% | 77.7% |

## K-RPO versus outcome-only

- revision_correct: +9.0 pp (95% CI [-1.5, +19.4]; Holm p=0.73).
- switch_success: +7.5 pp (95% CI [-1.5, +17.9]; Holm p=0.73).
- stale_original: -4.5 pp (95% CI [-11.9, +3.0]; Holm p=0.73).
- revision_action3_correct: +9.0 pp (95% CI [+0.0, +19.4]; Holm p=0.73).
- revision_final_and_pair_f1_ge_80: +7.5 pp (95% CI [-0.0, +16.4]; Holm p=0.73).
- revision_canonical_pair_f1: +2.1 pp (95% CI [+0.6, +3.7]).

The altered weights are synthetic and test contextual rule
execution; they are not proposed as clinically valid updates.

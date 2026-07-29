# Full-100 TCM clean-set evaluation

| Condition | Final joint | Weighted action-step | Complete trajectory | Schema valid |
|---|---:|---:|---:|---:|
| q14_base_clean | 21.0% | 48.7% | 6.0% | 100.0% |
| q14_outcome_clean | 28.0% | 56.2% | 6.0% | 68.0% |
| q14_krpo_clean | 38.0% | 69.7% | 21.0% | 100.0% |

## Paired comparisons

Holm correction is applied jointly to all five reported full-100 clean-set binary contrasts.

- krpo_minus_base_final_joint: +17.0 pp (95% CI [+7.0,+28.0]; Holm p=0.01331).
- krpo_minus_outcome_final_joint: +10.0 pp (95% CI [+0.0,+20.0]; Holm p=0.2266).
- krpo_minus_outcome_action_trajectory: +15.0 pp (95% CI [+8.0,+23.0]; Holm p=0.001373).
- outcome_minus_base_final_joint: +7.0 pp (95% CI [-3.0,+17.0]; Holm p=0.5299).
- outcome_minus_base_action_trajectory: +0.0 pp (95% CI [-6.0,+6.0]; Holm p=1).
- krpo_minus_outcome_per_case_action_step_exact: +14.3 pp (95% CI [+8.4,+20.2]).

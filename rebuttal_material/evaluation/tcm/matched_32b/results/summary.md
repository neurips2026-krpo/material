# Task12: independent full-100 Qwen3-32B evaluation

All metrics use the frozen Task7 exact evaluator, not the training reward.

| Condition | Final joint | Action-step exact | Action complete | Complete trajectory |
|---|---:|---:|---:|---:|
| q32_base_clean | 23.0% | 56.9% | 99.1% | 4.0% |
| q32_krpo_clean | 33.0% | 69.5% | 99.8% | 15.0% |

## Paired K-RPO minus Base comparisons

| Endpoint | Difference | 95% bootstrap CI | raw p | Holm p |
|---|---:|---:|---:|---:|
| final_joint_exact | 10.0% | [0.0%, 20.0%] | 0.07552 | 0.07552 |
| complete_trajectory_exact | 11.0% | [5.0%, 17.0%] | 0.0009766 | 0.001953 |

Action-step exact difference: 13.2% (paired bootstrap 95% CI [8.9%, 17.6%]).

Complete trajectory follows the frozen Task7/Task9 definition: every required categorical action step is exact. `final_and_action_joint` is retained separately as a stricter secondary diagnostic.

# Anonymous rebuttal artifact

This review-only artifact contains the training entry points, the K-RPO reward,
prompt template, frozen reward-independent evaluators, perturbation generators,
statistical scripts, public MedCalc experiment files, and the numerical outputs
reported in the author responses. Model weights and optimizer states are not
included.

## Correct training identity and sample accounting

Both submitted K-RPO checkpoints use Qwen3 backbones. The manuscript's Qwen2.5
reference is a naming/citation error, not a backbone change during rebuttal.
The training source contains 2,000 unique cases/reference trajectories. This is
distinct from the online sampled rollouts:

| Checkpoint | Export source | B | G | Updates | Sampled training rollouts |
|---|---|---:|---:|---:|---:|
| Qwen3-14B K-RPO | `global_step_100` | 256 | 10 | 100 | 256,000 |
| Qwen3-32B K-RPO | `global_step_75` | 256 | 10 | 75 | 192,000 |

Both submitted runs used `training/tcm_reward_v2.py`. The reward calls
Qwen3-Embedding-8B through an OpenAI-compatible embeddings API. The private
endpoint default was removed; set `EMBEDDING_API_URL` explicitly.

The two submitted checkpoints did not use an identical update schedule. The
reproduction scripts therefore expose their separately verified step counts.

## Key reward-independent TCM results

The primary evaluator performs deterministic exact matching over required typed
decisions. It does not import the training reward, call the embedding model, use
fuzzy/substring matching, or score the free-text conclusion.

| Backbone / method | Final joint | Complete typed trajectory |
|---|---:|---:|
| Qwen3-14B Base | 21% | 6% |
| Qwen3-14B outcome-only GRPO | 28% | 6% |
| Qwen3-14B K-RPO | 38% | 21% |
| Qwen3-32B Base | 23% | 4% |
| Qwen3-32B K-RPO | 33% | 15% |

For the 32B matched comparison, K-RPO improves complete typed trajectories by
11 percentage points (95% paired bootstrap CI +5 to +17; Holm-adjusted
McNemar p=.001953). Full paired outputs and metrics are included.

## Public Western rule-execution experiment

The public MedCalc-Bench Verified experiment contains 2,000 training cases and
380 evaluation cases. It is included as cross-domain portability evidence for
the configurable rule/typed-decision interface. The paired comparison files
report all four conditions (Base without rules, Base with rules, outcome-only
GRPO with rules, and K-RPO with rules), confidence intervals, and multiplicity
correction.

## Layout

- `data/`: submitted TCM train/validation/test JSONL and a dataset card.
- `training/`: sanitized K-RPO reward, prompt, checkpoint configs, and launch
  entry points.
- `evaluation/tcm/`: frozen exact evaluators, 14B/32B paired results, and
  knowledge perturbation experiments.
- `evaluation/medcalc/`: public data, reward/evaluator code, matched comparison,
  and synthetic current-rule stress test.
- `environment/`: relevant runtime package versions.



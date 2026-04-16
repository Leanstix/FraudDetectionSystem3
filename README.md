# Reply Mirror Agent Fraud Detection (The Truman Show)

This project is a complete, runnable, end-to-end Python solution for the Reply Mirror AI Agent Challenge on **THE TRUMAN SHOW** task.

It uses:
- **Reference dataset**: `The Truman Show - train.zip`
- **Target dataset**: `The Truman Show - validation.zip`

and produces:
- `outputs/the_truman_show_validation_submission.txt`
- `outputs/the_truman_show_validation_diagnostics.csv`

## What This Project Does

The pipeline detects suspicious transactions in the validation dataset by first learning behavioral priors from the train dataset.

Key properties:
- Agent-based architecture with specialist components
- Adaptive scoring with novelty and distribution-shift awareness
- Deterministic and configurable thresholding
- Strict ASCII submission formatting
- Optional Langfuse + OpenRouter integration with safe fallback when LLM is disabled or unavailable

## Why It Satisfies The Agent-Based Requirement

The system is not a single static script. It is a coordinated multi-agent workflow:

- `DataIngestionAgent`: loads and normalizes each dataset
- `BaselineBuilderAgent`: learns reference priors from train
- `EntityResolutionAgent`: links transactions to users/locations/communications
- `TransactionBehaviorAgent`: amount/profile anomalies and reference deltas
- `TemporalSequenceAgent`: burst and sequence anomalies with temporal priors
- `GeoSpatialAgent`: distance and geographic novelty risk
- `CommunicationRiskAgent`: thread-level communication risk with decay to transactions
- `NoveltyDriftAgent`: unsupervised anomaly/drift signals
- `FusionDecisionAgent`: weighted score fusion + deterministic bounded thresholding
- `SubmissionWriter`: strict output validation and writing

## Architecture Overview

Pipeline order:

1. Load + normalize reference dataset
2. Load + normalize target dataset
3. Build baseline from reference/train
4. Run entity resolution on target
5. Build target features with baseline delta features
6. Run specialist agents
7. Fuse scores and compute threshold
8. Validate and write submission
9. Write diagnostics CSV
10. Flush tracing

## Reference vs Target Scoring Strategy

The validation set is **not** scored in isolation.

`BaselineStore` fits on train and provides:
- sender norms (amount medians/MAD, seen methods/types)
- recipient norms
- sender-recipient pair priors
- geo priors
- hour/weekday priors
- communication keyword priors

Target transactions are scored against those references using features such as:
- `ref_sender_amount_robust_z`
- `ref_recipient_amount_robust_z`
- `pair_seen_in_reference`
- `payment_method_seen_by_sender_ref`
- `transaction_type_seen_by_sender_ref`
- `reference_hour_rarity`
- `reference_weekday_rarity`
- unseen indicators for method/type/location

## Langfuse + OpenRouter Integration

Integration follows the official tracking pattern:

- Session ID format: `{TEAM_NAME}-{ULID}`
- LangChain config metadata contains: `{"langfuse_session_id": session_id}`
- OpenRouter base URL: `https://openrouter.ai/api/v1`
- Langfuse flush at end of run
- LLM and tracing imports are lazy to avoid hard failures

If LLM dependencies are unavailable (or `--no-llm` is used), the pipeline still runs with heuristic communication scoring.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exact Truman Commands

Inspect both datasets:

```bash
python -m src.main inspect-pair --reference "The Truman Show - train.zip" --input "The Truman Show - validation.zip"
```

Create validation submission:

```bash
python -m src.main predict-pair --reference "The Truman Show - train.zip" --input "The Truman Show - validation.zip" --output "outputs/the_truman_show_validation_submission.txt"
```

Verbose mode:

```bash
python -m src.main predict-pair --reference "The Truman Show - train.zip" --input "The Truman Show - validation.zip" --output "outputs/the_truman_show_validation_submission.txt" --verbose
```

No LLM mode:

```bash
python -m src.main predict-pair --reference "The Truman Show - train.zip" --input "The Truman Show - validation.zip" --output "outputs/the_truman_show_validation_submission.txt" --no-llm
```

## Outputs Produced

- `outputs/the_truman_show_validation_submission.txt`
  - ASCII only
  - one transaction ID per line
  - deterministic ordering: `final_risk_score` desc, `transaction_id` asc
- `outputs/the_truman_show_validation_diagnostics.csv`
  - per-agent scores
  - final score
  - reasons and threshold context

## Assumptions

- No labels are used during scoring.
- Entity links are heuristic and confidence-weighted.
- Communication threads are analyzed at thread level, not per transaction.
- Optional LLM adds signal for ambiguous/suspicious threads only.

## Limitations

- Entity resolution can still miss links when IDs are sparse.
- Temporal and geo priors are lightweight and can be refined further.
- Current thresholding is deterministic and bounded, but not label-calibrated.

## Next Tuning Steps

1. Add adaptive weight tuning per dataset shift profile.
2. Add richer graph-based sender/recipient/entity risk propagation.
3. Improve communication-to-transaction causality linking windows.
4. Add robust drift monitors across rolling time slices.
5. Add post-run calibration diagnostics by feature cohort.

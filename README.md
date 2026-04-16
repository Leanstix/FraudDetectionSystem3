# Reply Mirror Multimodal Fraud Detection (Deus Ex)

This repository implements a complete, runnable, end-to-end, agent-based fraud detection system for the Reply Mirror AI Agent Challenge.

It is designed for train/validation pairing:
- Reference history: `Deus Ex - train.zip`
- Inference target: `Deus Ex - validation.zip`

It produces:
- `outputs/deus_ex_validation_submission.txt`
- `outputs/deus_ex_validation_diagnostics.csv`

## What The Project Does

The pipeline identifies suspicious validation transactions by combining specialized agent signals over:
- structured transactions
- user/entity profiles
- geospatial traces
- SMS and mail communications
- audio metadata (and optional transcription)

The system is adaptive: it learns baseline behavioral priors from TRAIN and scores deviations in VALIDATION.

## Why This Is Agent-Based

The solution is a cooperative multi-agent architecture, not a single static script.

Agents:
- `DataIngestionAgent`: load and normalize all modalities
- `EntityResolutionAgent`: probabilistic user/entity linking and context joins
- `TransactionBehaviorAgent`: amount and counterpart behavior anomalies
- `TemporalSequenceAgent`: velocity, burst, and timing rarity anomalies
- `GeoSpatialAgent`: residence/GPS distance and geo novelty
- `CommunicationRiskAgent`: thread-level comm risk scoring (heuristic-first, optional LLM)
- `AudioContextAgent`: speaker/time-linked audio risk scoring (metadata-first, optional LLM)
- `NoveltyDriftAgent`: unsupervised novelty/outlier analysis
- `FusionDecisionAgent`: weighted fusion + deterministic thresholding bounds
- `SubmissionWriter`: strict ASCII submission output validation/writing

## Architecture Overview

Pipeline order (`FraudPipeline.run_pair`):
1. Load + normalize TRAIN
2. Load + normalize VALIDATION
3. Entity resolution on VALIDATION
4. Build features using TRAIN as baseline, VALIDATION as target
5. Run specialist agents
6. Fuse agent scores + threshold
7. Validate + write submission TXT
8. Write diagnostics CSV
9. Flush tracing

## Train vs Validation Pairing

The model does not fit only on validation.

TRAIN contributes baseline priors for:
- sender/recipient amount behavior
- seen sender-recipient pairs
- seen payment methods and transaction types
- reference hour/weekday rarity
- geo signature context

VALIDATION transactions are scored as deviations from these priors plus multimodal contextual risk.

## Audio Handling

Audio support is explicit and robust:
- all `audio/*.mp3` files are indexed
- speaker and timestamp are inferred from filename when possible
- lightweight metadata extraction is attempted (duration, mime, size)
- speaker-to-user linking is heuristic and tolerant to imperfect names
- time-decayed proximity from audio events to transactions is used in scoring

Transcription:
- optional, controlled by env/config
- metadata-only mode is default
- missing transcription dependencies never break the pipeline

## Tracing

`TracingManager` supports Langfuse-compatible tracing when keys and dependencies are available.

- session ID format: `TEAM_NAME-ULID`
- safe fallback when tracing packages/keys are unavailable
- no secret values are printed

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exact Commands

Inspect pair:

```bash
python -m src.main inspect-pair --reference "Deus Ex - train.zip" --input "Deus Ex - validation.zip"
```

Generate submission:

```bash
python -m src.main predict-pair --reference "Deus Ex - train.zip" --input "Deus Ex - validation.zip" --output "outputs/deus_ex_validation_submission.txt"
```

Verbose mode:

```bash
python -m src.main predict-pair --reference "Deus Ex - train.zip" --input "Deus Ex - validation.zip" --output "outputs/deus_ex_validation_submission.txt" --verbose
```

No LLM mode:

```bash
python -m src.main predict-pair --reference "Deus Ex - train.zip" --input "Deus Ex - validation.zip" --output "outputs/deus_ex_validation_submission.txt" --no-llm
```

## Output Files

Submission TXT:
- plain ASCII only
- one `transaction_id` per line
- deterministic ordering: score desc, transaction ID asc

Diagnostics CSV:
- per-agent scores and reasons
- fusion score and threshold fields
- sorted by risk score desc then transaction ID asc

## Assumptions And Limitations

Assumptions:
- transaction schema follows challenge columns
- communications are semi-structured and parseable via headers/blobs
- audio filenames contain partial speaker/time clues in many cases

Limitations:
- entity linking is heuristic and may miss sparse aliases
- transcription is optional and disabled by default
- unsupervised anomaly signals can be sensitive to distribution drift

## Next Tuning Steps

1. Add graph-based risk propagation across entities and communications.
2. Improve speaker identity resolution using fuzzy matching and profile embeddings.
3. Add richer temporal drift monitors across rolling windows.
4. Tune fusion weights per dataset profile.
5. Add optional local ASR confidence-aware scoring when feasible.

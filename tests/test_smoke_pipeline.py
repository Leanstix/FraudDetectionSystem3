from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.pipeline.orchestrator import FraudPipeline


def test_pipeline_end_to_end_creates_outputs(tmp_path: Path):
    output_path = tmp_path / "the_truman_show_validation_submission.txt"
    settings = Settings.from_env_and_file()

    pipeline = FraudPipeline(
        settings=settings,
        reference_path="The Truman Show - train.zip",
        target_path="The Truman Show - validation.zip",
        output_path=str(output_path),
        dataset_name="the_truman_show",
        no_llm=True,
        verbose=False,
    )
    artifacts = pipeline.run()

    assert output_path.exists()
    assert Path(artifacts.diagnostics_path).exists()
    assert artifacts.submission.flagged_count > 0
    assert artifacts.submission.flagged_count < artifacts.submission.total_transactions

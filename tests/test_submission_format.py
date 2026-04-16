from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents.submission_writer import SubmissionWriter


def test_submission_output_constraints_and_ordering(tmp_path: Path):
    transactions_df = pd.DataFrame({"transaction_id": ["a", "b", "c", "d", "e"]})
    flagged_df = pd.DataFrame(
        {
            "transaction_id": ["d", "b", "a", "a", "c", "e"],
            "final_risk_score": [0.6, 0.9, 0.9, 0.91, 0.3, 0.1],
            "flagged": [True, True, True, True, False, False],
            "top_risk_reasons": ["r1", "r2", "r3", "r4", "r5", "r6"],
        }
    )

    out = tmp_path / "submission.txt"
    writer = SubmissionWriter()
    result = writer.run(flagged_df=flagged_df, transactions_df=transactions_df, output_path=str(out))

    text = out.read_text(encoding="ascii")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    assert lines
    assert len(lines) < len(transactions_df)
    assert set(lines).issubset(set(transactions_df["transaction_id"]))
    text.encode("ascii")
    assert lines == ["a", "b", "d"]
    assert result.flagged_count == 3
    assert all(isinstance(v, str) and v for v in result.transaction_ids)


def test_submission_ordering_is_deterministic(tmp_path: Path):
    transactions_df = pd.DataFrame({"transaction_id": ["x", "y", "z"]})
    flagged_df = pd.DataFrame(
        {
            "transaction_id": ["z", "y", "x"],
            "final_risk_score": [0.7, 0.7, 0.7],
            "flagged": [True, True, True],
            "top_risk_reasons": ["a", "b", "c"],
        }
    )

    writer = SubmissionWriter()
    out1 = tmp_path / "submission1.txt"
    out2 = tmp_path / "submission2.txt"

    writer.write(flagged_df, str(out1))
    writer.write(flagged_df, str(out2))

    assert out1.read_text(encoding="ascii") == out2.read_text(encoding="ascii")

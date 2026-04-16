from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents.base import BaseAgent
from src.types import SubmissionResult
from src.utils.io import write_ascii_lines
from src.utils.validation import assert_ascii_lines, assert_not_all


class SubmissionWriter(BaseAgent):
    name = "submission_writer"

    def validate(self, flagged_df: pd.DataFrame, transactions_df: pd.DataFrame) -> None:
        required_cols = {"transaction_id", "final_risk_score", "flagged"}
        missing = required_cols - set(flagged_df.columns)
        if missing:
            raise ValueError(f"Flagged dataframe missing columns: {sorted(missing)}")

        all_ids = set(transactions_df["transaction_id"].astype(str).tolist())
        flagged_ids = flagged_df.loc[flagged_df["flagged"], "transaction_id"].astype(str).tolist()

        if not set(flagged_ids).issubset(all_ids):
            raise ValueError("Submission includes unknown transaction IDs")

        assert_not_all(len(set(flagged_ids)), len(all_ids))

    def write(self, flagged_df: pd.DataFrame, output_path: str) -> SubmissionResult:
        ranked = (
            flagged_df.loc[flagged_df["flagged"]]
            .sort_values(["final_risk_score", "transaction_id"], ascending=[False, True])
            .drop_duplicates(subset=["transaction_id"])
        )
        ids = ranked["transaction_id"].astype(str).tolist()
        assert_ascii_lines(ids)

        write_ascii_lines(output_path, ids)

        total = len(flagged_df)
        count = len(ids)
        return SubmissionResult(
            output_path=output_path,
            flagged_count=count,
            total_transactions=total,
            flagged_percentage=(count / total) if total else 0.0,
            transaction_ids=ids,
        )

    def run(self, flagged_df: pd.DataFrame, transactions_df: pd.DataFrame, output_path: str) -> SubmissionResult:
        self.validate(flagged_df, transactions_df)
        return self.write(flagged_df, output_path)

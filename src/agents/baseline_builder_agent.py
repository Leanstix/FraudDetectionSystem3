from __future__ import annotations

import pandas as pd

from src.agents.base import BaseAgent
from src.data.baseline_store import BaselineStore
from src.types import BaselineArtifacts


class BaselineBuilderAgent(BaseAgent):
    name = "baseline_builder"

    def __init__(self, baseline_store: BaselineStore):
        self.baseline_store = baseline_store

    def run(self, reference_bundle: dict[str, pd.DataFrame]) -> BaselineArtifacts:
        self.baseline_store.fit(
            reference_transactions_df=reference_bundle["transactions"],
            reference_users_df=reference_bundle["users"],
            reference_locations_df=reference_bundle["locations"],
            reference_sms_df=reference_bundle["sms"],
            reference_mails_df=reference_bundle["mails"],
        )
        return self.baseline_store.export_artifacts()

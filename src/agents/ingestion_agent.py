from __future__ import annotations

import pandas as pd

from src.agents.base import BaseAgent
from src.data.loaders import DatasetLoader
from src.data.normalize import Normalizer


class DataIngestionAgent(BaseAgent):
    name = "data_ingestion"

    def __init__(self, loader: DatasetLoader, normalizer: Normalizer):
        self.loader = loader
        self.normalizer = normalizer

    def run(self) -> dict[str, pd.DataFrame]:
        transactions = self.normalizer.normalize_transactions(self.loader.load_transactions())
        users = self.normalizer.normalize_users(self.loader.load_users())
        locations = self.normalizer.normalize_locations(self.loader.load_locations())
        sms = self.normalizer.normalize_sms(self.loader.load_sms())
        mails = self.normalizer.normalize_mails(self.loader.load_mails())
        return {
            "transactions": transactions,
            "users": users,
            "locations": locations,
            "sms": sms,
            "mails": mails,
        }

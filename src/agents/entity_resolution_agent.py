from __future__ import annotations

import pandas as pd

from src.agents.base import BaseAgent
from src.data.entity_resolution import EntityResolver


class EntityResolutionAgent(BaseAgent):
    name = "entity_resolution"

    def __init__(self, resolver: EntityResolver):
        self.resolver = resolver

    def run(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        sms_df: pd.DataFrame,
        mails_df: pd.DataFrame,
    ) -> dict:
        linked = self.resolver.link_transactions_to_users(transactions_df, users_df)
        with_locations = self.resolver.attach_location_context(linked, locations_df)
        enriched = self.resolver.attach_communication_context(with_locations, sms_df, mails_df)

        diagnostics = {
            "sender_link_rate": float(enriched["sender_user_idx"].notna().mean()) if len(enriched) else 0.0,
            "recipient_link_rate": float(enriched["recipient_user_idx"].notna().mean()) if len(enriched) else 0.0,
            "avg_sender_link_score": float(enriched["sender_link_score"].mean()) if len(enriched) else 0.0,
            "avg_recipient_link_score": float(enriched["recipient_link_score"].mean()) if len(enriched) else 0.0,
        }

        profiles = self.resolver.build_entity_profiles(enriched, users_df, locations_df)
        return {
            "enriched_transactions_df": enriched,
            "entity_profiles": profiles,
            "linking_diagnostics": diagnostics,
        }

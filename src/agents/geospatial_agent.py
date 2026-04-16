from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.utils.math import minmax


class GeoSpatialAgent(BaseAgent):
    name = "geospatial"

    def run(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()
        dist_res = minmax(df.get("distance_from_residence_km", pd.Series(np.zeros(len(df)))).fillna(0.0).astype(float))
        dist_gps = minmax(df.get("distance_from_latest_gps_km", pd.Series(np.zeros(len(df)))).fillna(0.0).astype(float))
        geo_novelty = df.get("geo_novelty", pd.Series(np.zeros(len(df)))).fillna(0.0).astype(float).clip(0, 1)
        ref_geo_novelty = df.get("reference_geo_novelty", pd.Series(np.zeros(len(df)))).fillna(0.0).astype(float).clip(0, 1)
        unseen_geo = df.get("unseen_location_pattern_indicator", pd.Series(np.zeros(len(df)))).fillna(0.0).astype(float).clip(0, 1)

        score = (0.34 * dist_res + 0.27 * dist_gps + 0.16 * geo_novelty + 0.13 * ref_geo_novelty + 0.10 * unseen_geo).clip(0, 1)

        reasons = []
        for i in range(len(df)):
            parts = []
            if dist_res.iloc[i] > 0.75:
                parts.append("far_from_residence")
            if dist_gps.iloc[i] > 0.75:
                parts.append("far_from_latest_gps")
            if geo_novelty.iloc[i] > 0.5:
                parts.append("new_geo_context")
            if ref_geo_novelty.iloc[i] > 0.6:
                parts.append("geo_rare_vs_reference")
            if unseen_geo.iloc[i] > 0:
                parts.append("unseen_reference_location")
            reasons.append(";".join(parts[:3]) if parts else "geo_consistent")

        return pd.DataFrame(
            {
                "transaction_id": df["transaction_id"],
                "geospatial_score": score,
                "geospatial_reason": reasons,
            }
        )

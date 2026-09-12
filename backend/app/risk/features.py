"""CHAKRA Step 5: Feature Engineering Layer."""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.schemas.risk_scoring import RiskFeatures, FeatureAvailability
from app.risk.repository import RiskRepository

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts machine learning features deterministically."""

    def __init__(self, repository: RiskRepository):
        self.repo = repository

    async def extract_features(self, chain: str, address: str) -> RiskFeatures:
        """Extract all Step 5 features. Safely handles missing data."""
        stats = await self.repo.get_address_stats(chain, address)
        cluster_size = await self.repo.get_cluster_size(chain, address)
        velocity = await self.repo.get_inter_hop_velocity_hours(chain, address)
        
        # 1. Degree
        if stats and "degree" in stats:
            degree_feat = FeatureAvailability(available=True, value=float(stats["degree"]))
        else:
            degree_feat = FeatureAvailability(available=False, reason_if_missing="No transactions found")

        # 2. Transaction Volume
        if stats and "transaction_volume" in stats:
            vol_feat = FeatureAvailability(available=True, value=stats["transaction_volume"])
        else:
            vol_feat = FeatureAvailability(available=False, reason_if_missing="No transactions found")

        # 3. Hop distance to mixer
        # Currently, CHAKRA mixer detection (Step 3G) returns "insufficient_evidence".
        # We explicitly return unavailable instead of inventing a distance.
        mixer_feat = FeatureAvailability(
            available=False, 
            reason_if_missing="Mixer paths unavailable in current graph projection"
        )

        # 4. Time since first activity (in days)
        if stats and stats.get("first_activity_at"):
            first_at = stats["first_activity_at"]
            if first_at.tzinfo is None:
                first_at = first_at.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_since = (now - first_at).total_seconds() / 86400.0
            time_feat = FeatureAvailability(available=True, value=max(0.0, days_since))
        else:
            time_feat = FeatureAvailability(available=False, reason_if_missing="No timestamps found")

        # 5. Cluster size
        if cluster_size is not None:
            cluster_feat = FeatureAvailability(available=True, value=float(cluster_size))
        else:
            cluster_feat = FeatureAvailability(available=False, reason_if_missing="Address not in any cluster")

        # 6. Amount retention ratio = (incoming - outgoing) / incoming
        if stats and stats.get("total_incoming", 0.0) > 0:
            incoming = stats["total_incoming"]
            outgoing = stats["total_outgoing"]
            retention = (incoming - outgoing) / incoming
            ret_feat = FeatureAvailability(available=True, value=retention)
        else:
            ret_feat = FeatureAvailability(available=False, reason_if_missing="Zero or missing incoming volume")

        # 7. Inter-hop velocity
        if velocity is not None:
            vel_feat = FeatureAvailability(available=True, value=velocity)
        else:
            vel_feat = FeatureAvailability(available=False, reason_if_missing="Insufficient hops for velocity")

        return RiskFeatures(
            degree=degree_feat,
            transaction_volume=vol_feat,
            hop_distance_to_mixer=mixer_feat,
            time_since_first_activity=time_feat,
            cluster_size=cluster_feat,
            amount_retention_ratio=ret_feat,
            inter_hop_velocity=vel_feat
        )

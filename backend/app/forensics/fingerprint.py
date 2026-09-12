"""CHAKRA Step 6: Graph Fingerprinting."""
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class GraphFingerprint:
    """
    Generates a deterministic vector representation (embedding) 
    of an address's behavioral and structural properties.
    
    Dimensions:
    1. degree (normalized)
    2. transaction_volume (log-normalized)
    3. cluster_size (normalized)
    4. amount_retention_ratio ([-1, 1])
    5. inter_hop_velocity (normalized)
    6. temporal_density (txs per day)
    """

    @staticmethod
    def generate(
        degree: float,
        transaction_volume: float,
        cluster_size: float,
        amount_retention_ratio: float,
        inter_hop_velocity: float,
        days_active: float
    ) -> List[float]:
        """Compute the 6D fingerprint vector."""
        # 1. Degree (Sigmoid-like normalization centered around 10)
        norm_degree = 1.0 - (1.0 / (1.0 + (max(0, degree) / 10.0)))
        
        # 2. Volume (Log-normalized)
        norm_volume = np.log1p(max(0, transaction_volume)) / 20.0
        norm_volume = min(1.0, norm_volume)
        
        # 3. Cluster Size
        norm_cluster = 1.0 - (1.0 / (1.0 + (max(1, cluster_size) / 5.0)))
        
        # 4. Retention (-1 to 1 mapped to 0 to 1)
        retention = max(-1.0, min(1.0, amount_retention_ratio))
        norm_retention = (retention + 1.0) / 2.0
        
        # 5. Velocity (hours per hop -> higher score for faster)
        vel = max(0.1, inter_hop_velocity)
        norm_velocity = 1.0 / vel  # Fast = large value, slow = small value
        norm_velocity = min(1.0, norm_velocity)
        
        # 6. Temporal Density (txs / days_active)
        days = max(0.1, days_active)
        density = max(0, degree) / days
        norm_density = 1.0 - (1.0 / (1.0 + (density / 5.0)))
        
        vector = np.array([
            norm_degree,
            norm_volume,
            norm_cluster,
            norm_retention,
            norm_velocity,
            norm_density
        ], dtype=np.float32)
        
        # L2 Normalize the vector for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()

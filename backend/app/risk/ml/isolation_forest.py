"""CHAKRA Step 5: Isolation Forest wrapper."""
import logging
import os
import json
import pickle
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    IsolationForest = None
    SKLEARN_AVAILABLE = False


class IsolationForestModel:
    """Wrapper for the unsupervised anomaly detection model."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "isolation_forest.pkl")
        self.meta_path = os.path.join(model_dir, "isolation_forest_meta.json")
        self.model = None
        self.metadata = {}
        self.feature_names = [
            "degree", 
            "transaction_volume", 
            "time_since_first_activity", 
            "cluster_size", 
            "amount_retention_ratio", 
            "inter_hop_velocity"
        ]

    def is_trained(self) -> bool:
        return os.path.exists(self.model_path) and os.path.exists(self.meta_path)

    def load(self) -> bool:
        if not self.is_trained():
            return False
        if not SKLEARN_AVAILABLE:
            logger.warning("IsolationForest artifact exists, but 'sklearn' is missing. Cannot load.")
            return False
            
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Failed to load Isolation Forest: {e}")
            return False

    def predict(self, features_dict: Dict[str, float]) -> Optional[float]:
        """Run inference and return anomaly score normalized to 0-100."""
        if self.model is None:
            if not self.load():
                return None

        # Prepare feature vector (replace missing with median from training metadata if available, or 0)
        medians = self.metadata.get("feature_medians", {})
        x_array = []
        for fn in self.feature_names:
            val = features_dict.get(fn)
            if val is None:
                val = medians.get(fn, 0.0)
            x_array.append(float(val))
        
        try:
            X = np.array([x_array])
            # score_samples returns opposite of anomaly score (lower is more anomalous)
            # Typically range is [-0.5, 0.5]. 
            raw_score = self.model.score_samples(X)[0]
            
            # Normalize to 0-100 where 100 is most anomalous
            # Map [-1.0, 0.0] -> [100, 0] roughly
            norm_score = max(0.0, min(100.0, -raw_score * 100.0))
            return float(norm_score)
        except Exception as e:
            logger.error(f"Isolation Forest prediction failed: {e}")
            return None

"""CHAKRA Step 5: XGBoost Risk Model wrapper."""
import logging
import os
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Safe imports for environment without XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


class XGBoostRiskModel:
    """Wrapper for the XGBoost Step 5 Risk Model."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "xgboost_model.json")
        self.meta_path = os.path.join(model_dir, "xgboost_meta.json")
        self.model = None
        self.metadata = {}
        self.explainer = None
        self.feature_names = [
            "degree", 
            "transaction_volume", 
            "hop_distance_to_mixer", 
            "time_since_first_activity", 
            "cluster_size", 
            "amount_retention_ratio", 
            "inter_hop_velocity"
        ]

    def is_trained(self) -> bool:
        """Check if model artifact exists."""
        return os.path.exists(self.model_path) and os.path.exists(self.meta_path)

    def load(self) -> bool:
        """Load the model and metadata if available."""
        if not self.is_trained():
            return False
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost artifact exists, but 'xgboost' package is missing. Cannot load.")
            return False
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
            
            if SHAP_AVAILABLE:
                self.explainer = shap.TreeExplainer(self.model)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False

    def predict(self, features_dict: Dict[str, float]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """Run inference and return (score, shap_explanation)."""
        if self.model is None:
            if not self.load():
                return None, None

        # Prepare feature vector matching exact training order
        x_array = []
        for fn in self.feature_names:
            val = features_dict.get(fn)
            # Use NaN for missing values
            x_array.append(float(val) if val is not None else np.nan)
        
        dmatrix = xgb.DMatrix(np.array([x_array]), feature_names=self.feature_names)
        
        try:
            pred = self.model.predict(dmatrix)[0]
            # Convert raw prediction or probability to 0-100 scale based on training setup
            # Assuming the model outputs probability [0, 1]
            score = float(pred * 100.0)
            score = max(0.0, min(100.0, score))

            explanation = None
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(dmatrix)
                # Ensure it's a 1D array for a single instance
                if isinstance(shap_values, list): 
                    shap_values = shap_values[1] # For binary classification
                sv = shap_values[0] if len(np.array(shap_values).shape) > 1 else shap_values

                contributions = {self.feature_names[i]: float(sv[i]) for i in range(len(self.feature_names))}
                top_features = sorted(contributions.keys(), key=lambda k: abs(contributions[k]), reverse=True)[:3]
                
                explanation = {
                    "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value),
                    "feature_contributions": contributions,
                    "top_features": top_features
                }

            return score, explanation

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return None, None

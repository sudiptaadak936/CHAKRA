"""CHAKRA Step 5: Elliptic Dataset Loader."""
import os
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class EllipticDatasetLoader:
    """Loads and preprocesses the Elliptic dataset for CHAKRA ML pipelines."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
        self.edgelist_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
        self.features_path = os.path.join(data_dir, "elliptic_txs_features.csv")

    def validate_files(self) -> Tuple[bool, str]:
        """Check if all required files exist and return actionable message if missing."""
        missing = []
        if not os.path.exists(self.classes_path):
            missing.append("elliptic_txs_classes.csv")
        if not os.path.exists(self.edgelist_path):
            missing.append("elliptic_txs_edgelist.csv")
        if not os.path.exists(self.features_path):
            missing.append("elliptic_txs_features.csv")

        if missing:
            msg = (
                f"Missing Elliptic dataset files: {', '.join(missing)}. "
                "Please download the official Elliptic Data Set (e.g. from Kaggle) "
                f"and place the CSV files in: {self.data_dir}. "
                "The training pipeline cannot proceed without the real dataset."
            )
            return False, msg
        return True, "All files present."

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the data. Raises FileNotFoundError if missing."""
        valid, msg = self.validate_files()
        if not valid:
            raise FileNotFoundError(msg)

        logger.info("Loading Elliptic dataset (this may take a minute due to features file size)...")
        df_classes = pd.read_csv(self.classes_path)
        df_edges = pd.read_csv(self.edgelist_path)
        
        # Features file has no header. 
        # Col 0 is txId, Col 1 is time step, Col 2-166 are features.
        # But we only use a subset mapped to our custom CHAKRA features in Step 5.
        # For prototype simplicity we load it normally.
        df_features = pd.read_csv(self.features_path, header=None)
        
        return df_classes, df_edges, df_features

    def preprocess_for_xgboost(self, df_classes: pd.DataFrame, df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess dataset for XGBoost."""
        # 1=illicit, 2=licit, unknown=unknown
        # For supervised training, drop unknown
        df_labeled = df_classes[df_classes['class'] != 'unknown'].copy()
        
        # Map labels: 1 (illicit) -> 1, 2 (licit) -> 0
        df_labeled['label'] = df_labeled['class'].map({'1': 1, '2': 0, 1: 1, 2: 0})
        
        # Merge features
        # Column 0 of features is txId
        df_features.rename(columns={0: 'txId'}, inplace=True)
        merged = df_labeled.merge(df_features, on='txId', how='inner')
        
        y = merged['label']
        X = merged.drop(columns=['txId', 'class', 'label'])
        
        return X, y

import os
import sys
import json
import logging
import pickle
import argparse
import random
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch_geometric

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chakra_train")

def validate_model_input(X, expected_features):
    """Reusable function to validate CHAKRA inference inputs against trained schema."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    missing = [f for f in expected_features if f not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if X.isnull().values.any() or np.isinf(X.values).any():
        raise ValueError("Input contains NaN or Infinite values")
    return X[expected_features]

class EllipticDatasetLoader:
    def __init__(self, data_dir: str):
        self.classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
        self.edgelist_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
        self.features_path = os.path.join(data_dir, "elliptic_txs_features.csv")

    def load_data(self):
        for p in [self.classes_path, self.edgelist_path, self.features_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"CRITICAL ERROR: Missing dataset file {p}")
                
        df_classes = pd.read_csv(self.classes_path)
        df_edges = pd.read_csv(self.edgelist_path)
        df_features = pd.read_csv(self.features_path, header=None)
        if df_features.isnull().values.any():
            df_features.fillna(0, inplace=True)
        return df_classes, df_edges, df_features

def main():
    parser = argparse.ArgumentParser(description="CHAKRA Step 5 ML Training (Local/Fallback)")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to raw data directory")
    parser.add_argument("--output-dir", type=str, default="backend/models", help="Output directory")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "xgboost"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "isolation_forest"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "graph_ml"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "explainability"), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("==================================================")
    logger.info("ENVIRONMENT INFO:")
    logger.info(f"Python version: {sys.version.split(' ')[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"XGBoost version: {xgb.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("==================================================")

    loader = EllipticDatasetLoader(args.data_dir)
    logger.info("Loading dataset...")
    df_classes, df_edges, df_features = loader.load_data()
    
    df_labeled = df_classes[df_classes['class'] != 'unknown'].copy()
    df_labeled['label'] = df_labeled['class'].map({'1': 1, '2': 0, 1: 1, 2: 0})
    
    logger.info("Dataset Statistics:")
    logger.info(f"Total transactions: {len(df_classes)}")
    logger.info(f"Labeled transactions: {len(df_labeled)}")
    logger.info(f"  Illicit (Class 1): {sum(df_labeled['label'] == 1)}")
    logger.info(f"  Licit (Class 0): {sum(df_labeled['label'] == 0)}")
    logger.info(f"Total edges: {len(df_edges)}")
    
    logger.info("Calculating genuine graph features (vectorized)...")
    in_degrees = df_edges.iloc[:, 1].value_counts()
    out_degrees = df_edges.iloc[:, 0].value_counts()
    
    df_proto = pd.DataFrame({'txId': df_features[0]})
    df_proto['in_degree'] = df_proto['txId'].map(in_degrees).fillna(0).astype(int)
    df_proto['out_degree'] = df_proto['txId'].map(out_degrees).fillna(0).astype(int)
    df_proto['transaction_degree'] = df_proto['in_degree'] + df_proto['out_degree']
    
    logger.info("Mapping Elliptic proxy features (explicit schema separation)...")
    df_proto['elliptic_feature_1_time'] = df_features[1]
    df_proto['elliptic_feature_2'] = df_features[2]
    df_proto['elliptic_feature_3'] = df_features[3]
    df_proto['elliptic_feature_4'] = df_features[4]
    df_proto['elliptic_feature_5'] = df_features[5]
    df_proto['elliptic_feature_6'] = df_features[6]
    
    df_master = df_labeled.merge(df_proto, on='txId', how='inner')
    feature_names = [c for c in df_proto.columns if c != 'txId']
    
    logger.info("Splitting dataset (70% Train / 15% Val / 15% Test)...")
    txids = df_master['txId'].values
    y_labels = df_master['label'].values
    
    tx_train, tx_temp, _, y_temp = train_test_split(txids, y_labels, test_size=0.30, random_state=SEED, stratify=y_labels)
    tx_val, tx_test, _, _ = train_test_split(tx_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
    
    df_indexed = df_master.set_index('txId')
    X_train = df_indexed.loc[tx_train].drop(columns=['class', 'label'])
    y_train = df_indexed.loc[tx_train]['label']
    X_val = df_indexed.loc[tx_val].drop(columns=['class', 'label'])
    y_val = df_indexed.loc[tx_val]['label']
    X_test = df_indexed.loc[tx_test].drop(columns=['class', 'label'])
    y_test = df_indexed.loc[tx_test]['label']
    
    COMMON_METADATA = {
        "feature_schema_type": "ELLIPTIC_PROTOTYPE_PROXY",
        "feature_schema_description": "Features derived from the Elliptic Bitcoin dataset for prototype ML training. Elliptic features are proxy variables and are not equivalent to live CHAKRA multi-chain engineered features.",
        "limitations": "The current models are trained using the ELLIPTIC_PROTOTYPE_PROXY schema. A dedicated CHAKRA feature-adapter layer is required before these models can safely consume live blockchain data. Production deployment must validate that the live feature semantics match the training feature schema or retrain the models using genuinely engineered CHAKRA features.",
        "production_chakra_features_deferred": ["transaction_volume", "transaction_count", "wallet_age", "in_degree", "out_degree", "transaction_degree", "hop_distance_to_exchange", "hop_distance_to_mixer", "cluster_size", "amount_retention_ratio", "inter_hop_velocity", "bridge_hop_count", "mixer_exposure", "exchange_exposure", "cross_chain_hop_count"],
        "target_definition": "1 = illicit, 0 = licit",
        "random_seed": SEED,
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    logger.info("Training XGBoost Illicit Transaction Classifier...")
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'scale_pos_weight': ratio, 'max_depth': 6, 'eta': 0.1, 'seed': SEED, 'tree_method': 'hist'}
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds=10, verbose_eval=False)
    
    preds_prob_xgb = xgb_model.predict(dtest)
    preds_class_xgb = (preds_prob_xgb > 0.5).astype(int)
    xgb_metrics = {
        "accuracy": float(accuracy_score(y_test, preds_class_xgb)),
        "precision": float(precision_score(y_test, preds_class_xgb, zero_division=0)),
        "recall": float(recall_score(y_test, preds_class_xgb, zero_division=0)),
        "f1": float(f1_score(y_test, preds_class_xgb, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, preds_prob_xgb)),
    }
    
    xgb_out = os.path.join(args.output_dir, "xgboost")
    xgb_model.save_model(os.path.join(xgb_out, "xgboost_model.json"))
    with open(os.path.join(xgb_out, "xgboost_meta.json"), "w") as f:
        json.dump({"model_name": "XGBoost Illicit Transaction Classifier", "model_version": "1.0", "dataset_name": "Elliptic Bitcoin Dataset", "dataset_description": "200k Bitcoin transactions representing illicit vs licit patterns", "xgboost_version": xgb.__version__, "model_parameters": xgb_params, **COMMON_METADATA, "metrics": xgb_metrics, "feature_names": feature_names, "feature_count": len(feature_names), "training_samples": len(X_train), "validation_samples": len(X_val), "test_samples": len(X_test)}, f, indent=2)
        
    logger.info("Extracting SHAP feature importance...")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        shap_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': np.abs(shap_values).mean(axis=0)}).sort_values('mean_abs_shap', ascending=False)
        shap_df.to_csv(os.path.join(args.output_dir, "explainability", "xgboost_shap_summary.csv"), index=False)
    except Exception as e:
        logger.error(f"SHAP extraction failed gracefully: {e}")
        
    logger.info("Training Isolation Forest (Unsupervised)...")
    import sklearn
    # Calculate real contamination ratio from training data
    real_contamination = float(np.sum(y_train == 1) / len(y_train))
    
    iso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(n_estimators=100, contamination=real_contamination, random_state=SEED))
    ])
    iso_pipeline.fit(X_train)
    
    iso_out = os.path.join(args.output_dir, "isolation_forest")
    with open(os.path.join(iso_out, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso_pipeline, f)
        
    preds_iso = iso_pipeline.predict(X_test)
    preds_iso_mapped = np.where(preds_iso == -1, 1, 0)
    iso_metrics = {
        "accuracy": float(accuracy_score(y_test, preds_iso_mapped)),
        "precision": float(precision_score(y_test, preds_iso_mapped, zero_division=0)),
        "recall": float(recall_score(y_test, preds_iso_mapped, zero_division=0)),
        "f1": float(f1_score(y_test, preds_iso_mapped, zero_division=0))
    }
    with open(os.path.join(iso_out, "isolation_forest_meta.json"), "w") as f:
        json.dump({"model_name": "Isolation Forest Anomaly Detector", "model_version": "1.0", "dataset_name": "Elliptic Bitcoin Dataset", "training_type": "unsupervised", "sklearn_version": sklearn.__version__, **COMMON_METADATA, "contamination": real_contamination, "feature_names": feature_names, "feature_count": len(feature_names), "training_samples": len(X_train), "metrics": iso_metrics}, f, indent=2)

    logger.info("Constructing Graph for GAT...")
    node_mapping = {txid: i for i, txid in enumerate(df_proto['txId'].values)}
    x_tensor = torch.tensor(df_proto.drop(columns=['txId']).values, dtype=torch.float)
    
    edges_mapped = df_edges.copy()
    edges_mapped.iloc[:, 0] = edges_mapped.iloc[:, 0].map(node_mapping)
    edges_mapped.iloc[:, 1] = edges_mapped.iloc[:, 1].map(node_mapping)
    edges_mapped = edges_mapped.dropna().astype(int)
    src_v, dst_v = edges_mapped.iloc[:, 0].values, edges_mapped.iloc[:, 1].values
    edge_index = torch.tensor(np.array([np.concatenate([src_v, dst_v]), np.concatenate([dst_v, src_v])]), dtype=torch.long)
    
    y_tensor = torch.full((len(node_mapping),), -1, dtype=torch.float)
    train_mask = torch.zeros(len(node_mapping), dtype=torch.bool)
    val_mask = torch.zeros(len(node_mapping), dtype=torch.bool)
    test_mask = torch.zeros(len(node_mapping), dtype=torch.bool)
    
    def assign_mask(txids_arr, labels_series, mask_tensor):
        indices = [node_mapping[t] for t in txids_arr]
        mask_tensor[indices] = True
        y_tensor[indices] = torch.tensor(labels_series.values, dtype=torch.float)
        
    assign_mask(tx_train, y_train, train_mask)
    assign_mask(tx_val, y_val, val_mask)
    assign_mask(tx_test, y_test, test_mask)
    
    logger.info("This is a transductive GAT experiment. The graph structure and node features are available across the graph, while labels from validation/test nodes are excluded from the loss.")
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)
    
    class ChakraGAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.2):
            super().__init__()
            self.dropout = dropout
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

        def forward(self, x, edge_index, return_attention_weights=False):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if return_attention_weights:
                x, alpha1 = self.conv1(x, edge_index, return_attention_weights=True)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x, alpha2 = self.conv2(x, edge_index, return_attention_weights=True)
                return x, alpha1, alpha2
            else:
                x = F.elu(self.conv1(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.conv2(x, edge_index)
                return x
                
    logger.info("Training GAT (Transductive Experiment)...")
    gat_model = ChakraGAT(in_channels=x_tensor.shape[1], hidden_channels=16, out_channels=1, heads=2).to(device)
    pos_weight = torch.tensor([ratio], dtype=torch.float).to(device)
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_auc = 0
    best_epoch = 0
    best_state = None
    patience = 15
    patience_counter = 0
    max_epochs = 150
    
    for epoch in range(1, max_epochs + 1):
        gat_model.train()
        optimizer.zero_grad()
        out = gat_model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        gat_model.eval()
        with torch.no_grad():
            out = gat_model(data.x, data.edge_index)
            val_prob = torch.sigmoid(out[data.val_mask].squeeze()).cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()
            val_auc = roc_auc_score(val_y, val_prob)
            
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in gat_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
            
    logger.info("Restoring best model for final evaluation on unseen test mask...")
    gat_model.load_state_dict(best_state)
    gat_model.eval()
    with torch.no_grad():
        out = gat_model(data.x, data.edge_index)
        pred_prob_gat = torch.sigmoid(out[data.test_mask].squeeze()).cpu().numpy()
        pred_class_gat = (pred_prob_gat > 0.5).astype(int)
        y_true_gat = data.y[data.test_mask].cpu().numpy()
        gat_metrics = {
            "accuracy": float(accuracy_score(y_true_gat, pred_class_gat)),
            "precision": float(precision_score(y_true_gat, pred_class_gat, zero_division=0)),
            "recall": float(recall_score(y_true_gat, pred_class_gat, zero_division=0)),
            "f1": float(f1_score(y_true_gat, pred_class_gat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true_gat, pred_prob_gat))
        }
        
    gnn_out = os.path.join(args.output_dir, "graph_ml")
    torch.save(best_state, os.path.join(gnn_out, "graph_model.pt"))
    with open(os.path.join(gnn_out, "graph_meta.json"), "w") as f:
        json.dump({"model_name": "GAT Illicit Classifier", "model_version": "1.0", "dataset_name": "Elliptic Bitcoin Dataset", "transductive": True, "torch_version": torch.__version__, "torch_geometric_version": torch_geometric.__version__, **COMMON_METADATA, "architecture": "GATConv", "hidden_channels": 16, "attention_heads": 2, "dropout": 0.2, "learning_rate": 0.01, "weight_decay": 5e-4, "max_epochs": max_epochs, "best_epoch": best_epoch, "best_validation_auc": best_val_auc, "metrics": gat_metrics, "feature_names": feature_names, "feature_count": len(feature_names), "nodes_total": len(node_mapping), "nodes_train": int(train_mask.sum()), "nodes_validation": int(val_mask.sum()), "nodes_test": int(test_mask.sum())}, f, indent=2)
        
    logger.info("Extracting GAT Attention Weights (Safe Mode)...")
    gat_model.to('cpu')
    gat_model.eval()
    with torch.no_grad():
        try:
            _, _, alpha2 = gat_model(data.x.cpu(), data.edge_index.cpu(), return_attention_weights=True)
            attn_edge_index, attn_weights = alpha2
            attn_weights = attn_weights.squeeze().numpy()
            
            idx_to_tx = {v: k for k, v in node_mapping.items()}
            src_nodes = [idx_to_tx[idx.item()] for idx in attn_edge_index[0]]
            dst_nodes = [idx_to_tx[idx.item()] for idx in attn_edge_index[1]]
            
            attn_df = pd.DataFrame({'src': src_nodes, 'dst': dst_nodes, 'attention': attn_weights})
            attn_df = attn_df.sort_values('attention', ascending=False).head(50000)
            attn_df.to_csv(os.path.join(args.output_dir, "explainability", "gat_attention_edges.csv"), index=False)
        except Exception as e:
            logger.error(f"Attention extraction failed safely: {e}")
            
    logger.info("TRAINING PIPELINE COMPLETE.")

if __name__ == "__main__":
    main()

"""CHAKRA Step 5: Graph ML wrapper (GraphSAGE / GAT)."""
import logging
import os
import json
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    torch = None
    torch_geometric = None
    F = None
    Data = None
    GATConv = None
    SAGEConv = None
    PYG_AVAILABLE = False


if PYG_AVAILABLE:
    class ChakraGAT(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 1):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

        def forward(self, x, edge_index, return_attention_weights=False):
            if return_attention_weights:
                x, alpha1 = self.conv1(x, edge_index, return_attention_weights=True)
                x = F.elu(x)
                x, alpha2 = self.conv2(x, edge_index, return_attention_weights=True)
                return x, alpha1, alpha2
            else:
                x = F.elu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

class GraphRiskModel:
    """Wrapper for the GraphSAGE / GAT risk model."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "graph_model.pt")
        self.meta_path = os.path.join(model_dir, "graph_meta.json")
        self.model = None
        self.metadata = {}

    def is_trained(self) -> bool:
        return os.path.exists(self.model_path) and os.path.exists(self.meta_path)

    def load(self) -> bool:
        if not self.is_trained():
            return False
        if not PYG_AVAILABLE:
            logger.warning("Graph ML artifact exists, but 'torch_geometric' is missing. Cannot load.")
            return False
            
        try:
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
                
            in_channels = self.metadata.get("in_channels", 7)
            hidden_channels = self.metadata.get("hidden_channels", 16)
            
            self.model = ChakraGAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=1, heads=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Failed to load Graph Model: {e}")
            return False

    def _extract_2_hop_subgraph(self, features_dict: Dict[str, float]) -> Optional[Data]:
        """
        Simulate the extraction of a 2-hop subgraph around the target node.
        In a full production environment, this queries the Neo4j/Postgres database
        for 1-hop and 2-hop neighbors, returning their features and edges.
        Here we construct a mock small subgraph to demonstrate PyG inference.
        """
        if not PYG_AVAILABLE:
            return None
            
        # Create a tiny 3-node subgraph (Target, Neighbor 1, Neighbor 2)
        # We ensure it is a 2-hop subgraph by creating edges: 0 -> 1, 1 -> 2
        feature_list = [
            features_dict.get("degree", 0.0) or 0.0,
            features_dict.get("transaction_volume", 0.0) or 0.0,
            features_dict.get("hop_distance_to_mixer", 0.0) or 0.0,
            features_dict.get("time_since_first_activity", 0.0) or 0.0,
            features_dict.get("cluster_size", 0.0) or 0.0,
            features_dict.get("amount_retention_ratio", 0.0) or 0.0,
            features_dict.get("inter_hop_velocity", 0.0) or 0.0,
        ]
        
        # Target node
        x0 = torch.tensor(feature_list, dtype=torch.float32)
        
        # Mock neighbors with slightly perturbed features
        x1 = x0 * 0.9
        x2 = x0 * 1.1
        
        x = torch.stack([x0, x1, x2], dim=0)
        
        # Edges (2-hop subgraph): 0 -> 1, 1 -> 2, plus self loops
        edge_index = torch.tensor([
            [0, 1, 0, 1, 2],
            [1, 2, 0, 1, 2]
        ], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        return data

    def predict(self, features_dict: Dict[str, float]) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """Run inference on the 2-hop subgraph."""
        if self.model is None:
            if not self.load():
                return None, None
                
        if not PYG_AVAILABLE:
            return None, None

        # 1. Extract 2-hop subgraph for real-time inference
        data = self._extract_2_hop_subgraph(features_dict)
        if data is None:
            return None, None
            
        # 2. Run GNN model
        with torch.no_grad():
            out, alpha1, alpha2 = self.model(data.x, data.edge_index, return_attention_weights=True)
            # The output for the target node (index 0)
            logits = out[0].item()
            prob = torch.sigmoid(torch.tensor(logits)).item()
            score = prob * 100.0
            
            # 3. Explainability: Extract GAT attention weights
            # alpha2 is a tuple (edge_index, edge_weights)
            attn_edges, attn_weights = alpha2
            
            # Find which neighbors attended most to the target node
            # The target node is node 0 in our subgraph.
            # In PyG, GATConv computes message passing from source (j) to target (i).
            # alpha[i] represents attention of target i on source j.
            # So we look for edges where target is 0.
            target_edges_mask = attn_edges[1] == 0
            sources = attn_edges[0][target_edges_mask]
            weights = attn_weights[target_edges_mask].squeeze()
            
            # Ensure it's a 1D tensor even if only 1 element
            if weights.dim() == 0:
                weights = weights.unsqueeze(0)
                
            # Find the top contributing neighbor
            top_nodes = []
            if len(weights) > 0:
                top_idx = torch.argmax(weights).item()
                top_source_node = sources[top_idx].item()
                top_nodes.append(f"Subgraph Node {top_source_node}")
                
            gnn_exp = {
                "attention_weights_available": True,
                "top_contributing_nodes": top_nodes,
                "explanation_text": f"GNN attention highlights high interaction with {top_nodes[0] if top_nodes else 'itself'} in the 2-hop subgraph."
            }

        return score, gnn_exp

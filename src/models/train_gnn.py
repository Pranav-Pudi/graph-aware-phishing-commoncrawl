#!/usr/bin/env python3
"""
Unsupervised GraphSAGE training for node fingerprinting.

Trains a lightweight GraphSAGE model on a heterogeneous graph using link reconstruction loss.
Generates 32-dimensional structural fingerprints and clusters them into 100 networks via MiniBatchKMeans.
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class GNN_Fingerprinter(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv is well-suited for large graphs on CPUs
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE for node fingerprinting")
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to input Parquet with node features (final_gnn_features.parquet)"
    )
    parser.add_argument(
        "--edges",
        type=str,
        required=True,
        help="Path to edge list Parquet (graph_edges.parquet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/gnn_solver_results.parquet",
        help="Path to save final results with network_id and gnn_dim_* columns"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of positive edges to sample per epoch (default: 10000)"
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    edges_path = Path(args.edges)
    output_path = Path(args.output)

    if not features_path.is_file():
        logger.error(f"Features file not found: {features_path}")
        return
    if not edges_path.is_file():
        logger.error(f"Edges file not found: {edges_path}")
        return

    logger.info(f"Loading node features from: {features_path}")
    df = pd.read_parquet(features_path)

    logger.info(f"Loading edges from: {edges_path}")
    edges_df = pd.read_parquet(edges_path)

    # Prepare PyG Data object
    x = torch.tensor(df['topic'].values.reshape(-1, 1), dtype=torch.float)
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    logger.info(f"Graph initialized: {data.num_nodes} nodes, {data.num_edges} edges")

    # Initialize model (32-dim output)
    model = GNN_Fingerprinter(in_channels=1, hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop: unsupervised link reconstruction
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)

        # Sample positive edges
        sample_idx = torch.randint(0, data.num_edges, (args.batch_size,))
        s, t = data.edge_index[0, sample_idx], data.edge_index[1, sample_idx]

        # Loss: high cosine similarity for connected nodes
        pos_loss = -torch.log(torch.sigmoid((z[s] * z[t]).sum(dim=-1)) + 1e-15).mean()
        pos_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Structural Loss: {pos_loss.item():.4f}")

    # Extract final fingerprints
    logger.info("Generating final node fingerprints...")
    model.eval()
    with torch.no_grad():
        z_final = model(data.x, data.edge_index).numpy()

    # Cluster into 100 networks
    logger.info("Clustering networks using MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)
    network_labels = kmeans.fit_predict(z_final)

    # Save results
    df['network_id'] = network_labels
    for i in range(32):
        df[f'gnn_dim_{i}'] = z_final[:, i]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Training complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Build graph from processed web page features for GNN input.
Edges based on shared identity (GA IDs), semantic similarity (topics), and structural template match.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

# ────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("graph_build.log")]
)
logger = logging.getLogger(__name__)

def build_identity_edges(df: pd.DataFrame) -> list:
    """Connect pages sharing the same Google Analytics ID (clique, but sampled)."""
    logger.info("Building identity edges (shared GA IDs)...")
    edges = []
    groups = df.groupby('ga_id').groups
    for ga_id, indices in tqdm(groups.items(), desc="Identity groups"):
        if len(indices) > 1 and ga_id is not None:
            indices = list(indices)
            # Full clique is O(n²) — sample pairs if group is large
            if len(indices) > 100:
                np.random.seed(42)
                idx_pairs = np.random.choice(indices, size=(500, 2), replace=True)
                for i, j in idx_pairs:
                    if i != j:
                        edges.append((min(i, j), max(i, j), 0))  # 0 = identity
            else:
                for i in indices:
                    for j in indices:
                        if i < j:
                            edges.append((i, j, 0))
    return edges

def build_semantic_edges(df: pd.DataFrame, min_shared_topic_size=5) -> list:
    """Connect pages in the same topic cluster (sampled if cluster is large)."""
    logger.info("Building semantic edges (shared topics)...")
    edges = []
    groups = df.groupby('topic').groups
    for topic, indices in tqdm(groups.items(), desc="Semantic groups"):
        if len(indices) >= min_shared_topic_size and topic != -1:  # skip noise
            indices = list(indices)
            # Limit to reasonable number of edges per cluster
            max_edges = 5000
            if len(indices) > 100:
                np.random.seed(42)
                idx_pairs = np.random.choice(indices, size=(max_edges, 2), replace=True)
                for i, j in idx_pairs:
                    if i != j:
                        edges.append((min(i, j), max(i, j), 1))  # 1 = semantic
            else:
                for i in indices:
                    for j in indices:
                        if i < j:
                            edges.append((i, j, 1))
    return edges

def build_structural_edges(df: pd.DataFrame, min_group_size=3) -> list:
    """Connect pages with identical tag fingerprint strings."""
    logger.info("Building structural edges (identical tag fingerprints)...")
    df['tag_str'] = df['tag_counts_json'].astype(str)
    groups = df.groupby('tag_str').groups
    edges = []
    for tag_str, indices in tqdm(groups.items(), desc="Structural groups"):
        if len(indices) >= min_group_size:
            indices = list(indices)
            for i in indices:
                for j in indices:
                    if i < j:
                        edges.append((i, j, 2))  # 2 = structural
    return edges

def main():
    parser = argparse.ArgumentParser(description="Build graph from processed features")
    parser.add_argument("--input", type=str, required=True, help="Path to final_gnn_features.parquet")
    parser.add_argument("--output", type=str, default="outputs/graph_edges.parquet")
    args = parser.parse_args()

    logger.info("Starting graph construction")

    df = pd.read_parquet(args.input).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} nodes")

    # Build edges
    identity_edges = build_identity_edges(df)
    semantic_edges = build_semantic_edges(df)
    structural_edges = build_structural_edges(df)

    all_edges = identity_edges + semantic_edges + structural_edges

    logger.info(f"Total edges: {len(all_edges)}")
    logger.info(f"Identity edges: {len(identity_edges)}")
    logger.info(f"Semantic edges: {len(semantic_edges)}")
    logger.info(f"Structural edges: {len(structural_edges)}")

    # Convert to edge list DataFrame
    edge_df = pd.DataFrame(all_edges, columns=['source', 'target', 'relation_type'])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    edge_df.to_parquet(output_path, compression="snappy", index=False)
    logger.info(f"Graph edge list saved: {output_path}")

    # Optional: Save sparse adjacency matrix (for GNN libraries)
    row = edge_df['source'].values
    col = edge_df['target'].values
    data = np.ones(len(edge_df))
    adj = coo_matrix((data, (row, col)), shape=(len(df), len(df)))
    np.savez(output_path.with_suffix('.npz'), row=adj.row, col=adj.col, data=adj.data, shape=adj.shape)
    logger.info(f"Sparse adjacency saved: {output_path.with_suffix('.npz')}")

if __name__ == "__main__":
    main()

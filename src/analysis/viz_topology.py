#!/usr/bin/env python3
"""
Visualize Network 99 topology (internal or external connectivity) using NetworkX and matplotlib.
Uses Agg backend for headless environments (HPC, servers).
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Visualize Network 99 connectivity graph")
    parser.add_argument(
        "--gnn-results",
        type=str,
        required=True,
        help="Path to gnn_solver_results.parquet (contains 'network_id' and index)"
    )
    parser.add_argument(
        "--edges",
        type=str,
        required=True,
        help="Path to graph_edges.parquet (source, target columns)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/viz/network_99_connectivity.png",
        help="Output PNG file path (default: outputs/viz/network_99_connectivity.png)"
    )
    parser.add_argument(
        "--sample-edges",
        type=int,
        default=500,
        help="Maximum number of edges to sample for visualization (default: 500)"
    )
    args = parser.parse_args()

    gnn_path = Path(args.gnn_results)
    edges_path = Path(args.edges)
    output_path = Path(args.output)

    if not gnn_path.is_file():
        logger.error(f"GNN results file not found: {gnn_path}")
        return
    if not edges_path.is_file():
        logger.error(f"Edges file not found: {edges_path}")
        return

    logger.info(f"Loading GNN results from: {gnn_path}")
    df = pd.read_parquet(gnn_path)

    logger.info(f"Loading edges from: {edges_path}")
    edges = pd.read_parquet(edges_path)

    # Identify Network 99 nodes (using DataFrame index)
    net99_nodes = df[df['network_id'] == 99].index.tolist()
    node_set = set(net99_nodes)
    logger.info(f"Found {len(node_set)} nodes in Network 99")

    # Filter internal edges first
    mask_internal = edges['source'].isin(node_set) & edges['target'].isin(node_set)
    internal_edges = edges[mask_internal]

    if len(internal_edges) > 0:
        logger.info(f"Found {len(internal_edges)} internal edges. Sampling up to {args.sample_edges}.")
        plot_edges = internal_edges.head(args.sample_edges)
        title_text = "Network 99 Internal Connectivity (Sample)"
    else:
        logger.info("No internal edges found. Falling back to all edges touching Network 99.")
        mask_external = edges['source'].isin(node_set) | edges['target'].isin(node_set)
        plot_edges = edges[mask_external].head(args.sample_edges)
        title_text = "Network 99 External Connectivity (Sample)"

    if len(plot_edges) == 0:
        logger.error("No edges available for visualization. Check data integrity.")
        return

    # Build graph
    G = nx.from_pandas_edgelist(plot_edges, 'source', 'target')

    # Plot
    plt.figure(figsize=(15, 15))
    plt.gca().set_facecolor('#f0f0f0')

    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#2c3e50', width=1.5)

    node_colors = ['#e67e22' if n in node_set else '#3498db' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos,
                           node_size=300,
                           node_color=node_colors,
                           edgecolors='black',
                           linewidths=1.5)

    plt.title(f"GNN Solver: {title_text}", fontsize=22, fontweight='bold')
    plt.axis('off')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor='#f0f0f0', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Graph visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
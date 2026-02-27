#!/usr/bin/env python3
"""
Generate key visualizations from the GNN pipeline results:
- Topic distribution bar plot (top semantic clusters)
- Pie chart of top networks by size
- Subgraph sample of Network 99 connectivity

Uses Agg backend for headless environments (HPC, servers).
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate pipeline result visualizations")
    parser.add_argument(
        "--gnn-results",
        type=str,
        required=True,
        help="Path to gnn_solver_results.parquet (contains 'network_id', 'topic', etc.)"
    )
    parser.add_argument(
        "--topic-dict",
        type=str,
        required=True,
        help="Path to topic_dictionary.csv (from BERTopic inspection)"
    )
    parser.add_argument(
        "--edges",
        type=str,
        required=True,
        help="Path to graph_edges.parquet (source, target columns)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/viz",
        help="Directory to save plots (default: outputs/viz)"
    )
    parser.add_argument(
        "--sample-edges",
        type=int,
        default=200,
        help="Maximum edges to sample for Network 99 topology plot (default: 200)"
    )
    args = parser.parse_args()

    gnn_path = Path(args.gnn_results)
    topic_path = Path(args.topic_dict)
    edges_path = Path(args.edges)
    viz_dir = Path(args.output_dir)

    if not gnn_path.is_file():
        logger.error(f"GNN results file not found: {gnn_path}")
        return
    if not topic_path.is_file():
        logger.error(f"Topic dictionary not found: {topic_path}")
        return
    if not edges_path.is_file():
        logger.error(f"Edges file not found: {edges_path}")
        return

    logger.info(f"Loading GNN results from: {gnn_path}")
    df = pd.read_parquet(gnn_path)

    logger.info(f"Loading topic dictionary from: {topic_path}")
    topics = pd.read_csv(topic_path)

    logger.info(f"Loading edges from: {edges_path}")
    edges = pd.read_parquet(edges_path)

    viz_dir.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────
    # Plot 1: Topic Distribution (Top 15)
    # ────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    top_topics = topics.head(15)
    sns.barplot(data=top_topics, x='Count', y='Name', palette='viridis')
    plt.title('Top 15 Semantic Clusters in the Vietnamese Web (BERT)')
    plt.tight_layout()
    plt.savefig(viz_dir / 'topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: topic_distribution.png")

    # ────────────────────────────────────────────────
    # Plot 2: Network Size Hierarchy (Pie Chart)
    # ────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    net_counts = df['network_id'].value_counts().head(10)
    plt.pie(net_counts, labels=net_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('pastel'))
    plt.title('Top 10 Discovered Networks (GNN Solver)')
    plt.savefig(viz_dir / 'network_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: network_pie.png")

    # ────────────────────────────────────────────────
    # Plot 3: Network 99 Topology Sample
    # ────────────────────────────────────────────────
    plt.figure(figsize=(12, 12))
    net99_nodes = df[df['network_id'] == 99].index[:50]  # small sample for clarity
    mask = edges['source'].isin(net99_nodes) & edges['target'].isin(net99_nodes)
    sample_edges = edges[mask].head(args.sample_edges)

    if len(sample_edges) == 0:
        logger.warning("No edges found in Network 99 sample. Skipping topology plot.")
    else:
        G = nx.from_pandas_edgelist(sample_edges, 'source', 'target')
        pos = nx.spring_layout(G, k=0.5)

        nx.draw(G, pos,
                node_size=40,
                node_color='#1f77b4',
                edge_color='#cccccc',
                alpha=0.6,
                with_labels=False)

        plt.title('GNN Topology: A Slice of Network 99 (24H Group)')
        plt.savefig(viz_dir / 'network_99_topology.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved: network_99_topology.png")

    logger.info("All visualizations completed.")


if __name__ == "__main__":
    main()


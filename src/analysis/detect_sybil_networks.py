#!/usr/bin/env python3
"""
Rank networks by diversity of Google Analytics IDs as a sybil/coordination signal.

Counts unique GA IDs per network cluster and highlights the most suspicious ones.
Also reports the dominant topic in the top-ranked network.
"""

import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Rank networks by GA ID diversity (sybil signal)")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to gnn_solver_results.parquet (must contain 'network_id' and 'ga_id')"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top suspicious networks to display (default: 10)"
    )
    args = parser.parse_args()

    input_path = args.input
    logger.info(f"Reading GNN results from: {input_path}")

    df = pd.read_parquet(input_path)

    # Filter rows that actually have a GA ID
    has_ga = df.dropna(subset=['ga_id'])

    # Count unique GA IDs per network, sorted descending
    sybil_report = has_ga.groupby('network_id')['ga_id'].nunique().sort_values(ascending=False)

    print(f"\nTOP {args.top_n} SUSPICIOUS NETWORKS (Network ID | Unique GA IDs | Pages in Network):")
    for net_id, unique_ga in sybil_report.head(args.top_n).items():
        page_count = len(df[df['network_id'] == net_id])
        print(f"  {net_id:>3}  →  {unique_ga:>6} unique GA IDs   ({page_count:>6} pages)")

    # Most suspicious network summary
    if not sybil_report.empty:
        top_net = sybil_report.index[0]
        top_pages = len(df[df['network_id'] == top_net])
        top_ga = sybil_report.iloc[0]
        dominant_topic = df[df['network_id'] == top_net]['topic'].mode().values[0]

        print(f"\nMost suspicious network: {top_net}")
        print(f"  - Pages: {top_pages:,}")
        print(f"  - Unique GA IDs: {top_ga:,}")
        print(f"  - Dominant topic: {dominant_topic}")
        print("  → Strong candidate for coordinated / sybil-like behavior.")
    else:
        print("\nNo networks with GA IDs found.")

if __name__ == "__main__":
    main()


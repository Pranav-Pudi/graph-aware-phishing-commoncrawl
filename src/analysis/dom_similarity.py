#!/usr/bin/env python3
"""
Compute pairwise Jaccard similarity on HTML tag sets for pages in Network 99.

Samples pages from the GNN results, extracts tag names using BeautifulSoup,
computes similarity matrix, and saves top similar pairs (including cross-domain).
Uses tqdm for progress and logging for visibility.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def get_tag_set(path: Path) -> set:
    """Extract set of tag names from HTML file."""
    if not path.is_file():
        logger.warning(f"File not found: {path}")
        return set()
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'lxml')
        return {tag.name for tag in soup.find_all() if tag.name}
    except Exception as e:
        logger.warning(f"Failed parsing {path}: {e}")
        return set()

def main():
    parser = argparse.ArgumentParser(description="Compute tag-set Jaccard similarity for Network 99 pages")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to gnn_solver_results.parquet (must contain 'network_id' and 'path')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/suspicious_network99_dom_pairs.csv",
        help="Path to save CSV of similar pairs (default: outputs/suspicious_network99_dom_pairs.csv)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of pages to sample from Network 99 (default: 50)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading GNN results from: {input_path}")
    df = pd.read_parquet(input_path)

    suspicious = df[df['network_id'] == 99].sample(n=args.sample_size, random_state=42)
    logger.info(f"Sampling {len(suspicious)} pages from Network 99")

    tag_sets = []
    paths = []
    for _, row in tqdm(suspicious.iterrows(), total=len(suspicious), desc="Extracting tags"):
        tag_set = get_tag_set(Path(row['path']))
        tag_sets.append(tag_set)
        paths.append(row['path'])

    if not tag_sets:
        logger.error("No valid tag sets extracted. Check input paths or file access.")
        return

    # Convert sets to binary vectors for Jaccard
    all_tags = sorted(set().union(*tag_sets))
    binary = np.array([[1 if t in s else 0 for t in all_tags] for s in tag_sets])

    # Pairwise Jaccard distance → similarity
    dist_matrix = pairwise_distances(binary, metric='jaccard')
    sim_matrix = 1 - dist_matrix

    # Collect pairs
    pairs = []
    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'path1': paths[i],
                'path2': paths[j],
                'similarity': sim_matrix[i, j],
                'common_tags': len(tag_sets[i] & tag_sets[j])
            })

    pairs_df = pd.DataFrame(pairs).sort_values('similarity', ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(output_path, index=False)
    logger.info(f"Top similar pairs saved to: {output_path}")

    print("\nTop 10 most similar pairs:")
    print(pairs_df.head(10)[['path1', 'path2', 'similarity', 'common_tags']])

    print(f"\nAverage similarity within sample: {sim_matrix.mean():.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute DOM tag-set similarity (Jaccard) for Network 99 pages.
"""

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from sklearn.metrics import pairwise_distances
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tag_set(path: str) -> set:
    path = Path(path)
    if not path.is_file():
        return set()
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'lxml')
        return set(tag.name for tag in soup.find_all() if tag.name)
    except Exception as e:
        logger.warning(f"Failed {path}: {e}")
        return set()

def main():
    df = pd.read_parquet('/hpctmp/e1430653/project/outputs/gnn_solver_results.parquet')
    suspicious = df[df['network_id'] == 99].sample(50, random_state=42)  # adjust size

    logger.info(f"Sampling {len(suspicious)} pages from Network 99")

    tag_sets = []
    paths = []
    for _, row in tqdm(suspicious.iterrows(), total=len(suspicious)):
        tag_set = get_tag_set(row['path'])
        tag_sets.append(tag_set)
        paths.append(row['path'])

    # Convert sets to binary vectors for Jaccard
    all_tags = sorted(set().union(*tag_sets))
    binary = np.array([[1 if t in s else 0 for t in all_tags] for s in tag_sets])

    # Pairwise Jaccard distance (0 = identical, 1 = no overlap)
    dist_matrix = pairwise_distances(binary, metric='jaccard')

    # Convert distance to similarity (1 - distance)
    sim_matrix = 1 - dist_matrix

    # Save top similar pairs
    pairs = []
    n = len(paths)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append({
                'path1': paths[i],
                'path2': paths[j],
                'similarity': sim_matrix[i,j],
                'common_tags': len(tag_sets[i] & tag_sets[j])
            })

    pairs_df = pd.DataFrame(pairs).sort_values('similarity', ascending=False)
    pairs_df.to_csv('/hpctmp/e1430653/suspicious_network99_dom_pairs.csv', index=False)
    logger.info(f"Top 10 most similar pairs saved to suspicious_network99_dom_pairs.csv")

    print("\nTop 10 most similar pairs:")
    print(pairs_df.head(10)[['path1', 'path2', 'similarity', 'common_tags']])

    print(f"\nAverage similarity within sample: {sim_matrix.mean():.3f}")

if __name__ == "__main__":
    main()

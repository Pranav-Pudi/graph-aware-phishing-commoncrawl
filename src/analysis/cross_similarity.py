#!/usr/bin/env python3
"""
Cross-domain sibling similarity analysis for Network 99.

Extracts domains from encoded paths, filters for cross-domain high-similarity pairs,
prints top siblings and average cross-domain similarity, and saves filtered results.
"""

import argparse
import logging
import re
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def extract_domain(path: str) -> str:
    """Extract domain name from encoded path (between 'https___' and next '_')."""
    match = re.search(r'https___([^_]+)', str(path))
    return match.group(1) if match else "unknown"

def main():
    parser = argparse.ArgumentParser(description="Analyze cross-domain similarity pairs")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV (e.g., suspicious_network99_dom_pairs.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="network99_siblings.csv",
        help="Path to save filtered cross-domain results (default: current directory)"
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    logger.info(f"Reading input CSV: {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Extracting domains from paths...")
    df['dom1'] = df['path1'].apply(extract_domain)
    df['dom2'] = df['path2'].apply(extract_domain)

    logger.info("Filtering for cross-domain pairs...")
    cross_df = df[df['dom1'] != df['dom2']].copy()

    print("=== CROSS-DOMAIN SIBLING ANALYSIS ===")
    if cross_df.empty:
        print("No cross-domain pairs found in this sample. Try a larger sample in dom_similarity.py")
    else:
        top_siblings = cross_df.sort_values('similarity', ascending=False).head(10)
        print(top_siblings[['dom1', 'dom2', 'similarity', 'common_tags']])
        print(f"\nAverage Cross-Domain Similarity: {cross_df['similarity'].mean():.3f}")

    logger.info(f"Saving filtered results to: {output_path}")
    cross_df.to_csv(output_path, index=False)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()


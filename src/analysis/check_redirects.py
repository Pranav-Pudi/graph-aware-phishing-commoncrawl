#!/usr/bin/env python3
"""
Bulk redirect checker for Network 99 pages.
Follows HTTP redirects and reports final URL and status.
"""

import argparse
import logging
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def check_redirect(url: str, timeout=10, max_redirects=10) -> dict:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        return {
            'original_url': url,
            'final_url': resp.url,
            'status_code': resp.status_code,
            'redirect_count': len(resp.history),
            'redirect_chain': [h.url for h in resp.history],
            'error': None
        }
    except requests.exceptions.RequestException as e:
        return {
            'original_url': url,
            'final_url': None,
            'status_code': None,
            'redirect_count': 0,
            'redirect_chain': [],
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Check redirects in Network 99")
    parser.add_argument("--input", type=str, required=True, help="Path to gnn_solver_results.parquet")
    parser.add_argument("--output", type=str, default="outputs/network99_redirects.csv")
    parser.add_argument("--sample", type=int, default=200, help="Number of pages to check")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    suspicious = df[df['network_id'] == 99].sample(n=args.sample, random_state=42)

    logger.info(f"Checking redirects for {len(suspicious)} sample pages from Network 99")

    results = []
    for _, row in tqdm(suspicious.iterrows(), total=len(suspicious)):
        url = row['path']  # adjust if your URL column is named differently
        result = check_redirect(url)
        result['original_id'] = row.name  # row index
        result['ga_id'] = row.get('ga_id')
        results.append(result)

    result_df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Redirect results saved to {output_path}")

    # Quick summary
    print("\nRedirect summary:")
    print(result_df['redirect_count'].value_counts().sort_index())
    print("\nTop 10 final destinations (most frequent):")
    print(result_df['final_url'].value_counts().head(10))

if __name__ == "__main__":
    main()

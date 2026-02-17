#!/usr/bin/env python3
"""
Parallel HTML feature extraction for DOM template clustering.
Extracts text, structural metadata, and lightweight graph stats.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from functools import partial
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import re
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("extract.log")]
)
logger = logging.getLogger(__name__)

def extract_features(path: Path, label: str) -> dict | None:
    """Process one HTML file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            html = f.read()

        soup = BeautifulSoup(html, 'lxml')

        # Tracking IDs
        ga_ids = re.findall(r'UA-\d+-\d+', html)
        pub_ids = re.findall(r'pub-\d+', html)

        # External scripts sample
        scripts = [s.get('src', '').strip() for s in soup.find_all('script', src=True)][:5]

        # Cleaned text
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)[:1200]

        # Structural counts
        tag_counts = {tag: len(soup.find_all(tag)) for tag in ["div", "a", "p", "li", "form", "input", "button"]}

        return {
            "id": path.name,
            "source": label,
            "path": str(path),
            "text": text,
            "ga_id": ga_ids[0] if ga_ids else None,
            "pub_id": pub_ids[0] if pub_ids else None,
            "script_samples": "|".join(scripts) if scripts else None,
            "file_size_bytes": path.stat().st_size,
            "tag_counts": json.dumps(tag_counts)
        }
    except Exception as e:
        logger.warning(f"Failed {path}: {str(e)[:200]}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vn-dir", type=str, required=True)
    parser.add_argument("--non-vn-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="features.parquet")
    parser.add_argument("--nproc", type=int, default=24)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    logger.info(f"Starting extraction with {args.nproc} processes")

    tasks = []
    vn_dir = Path(args.vn_dir)
    non_vn_dir = Path(args.non_vn_dir)

    vn_files = list(vn_dir.glob("*.html"))
    non_vn_files = list(non_vn_dir.glob("*.html"))

    if args.sample:
        vn_files = vn_files[:args.sample]
        non_vn_files = non_vn_files[:args.sample]

    tasks.extend((p, "vn") for p in vn_files)
    tasks.extend((p, "non_vn") for p in non_vn_files)

    logger.info(f"Processing {len(tasks)} files")

    with mp.Pool(args.nproc) as pool:
        results = list(tqdm(pool.imap(partial(extract_features), tasks), total=len(tasks)))

    df = pd.DataFrame([r for r in results if r])
    df.to_parquet(args.output, compression="snappy", index=False)

    logger.info(f"Processed {len(df)} files → {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
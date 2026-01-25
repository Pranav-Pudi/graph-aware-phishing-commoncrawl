"""
fetch_warc_range.py

Fetches exact byte ranges from Common Crawl WARC files using anonymous S3 access,
extracts HTML content, and saves it locally for Vietnamese phishing/benign analysis.

Dependencies (add to requirements.txt if missing):
- boto3
- botocore
- warcio
- pandas
- tqdm

Run on NUS HPC login node (internet required for S3).
"""

import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from warcio.archiveiterator import ArchiveIterator
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import sys
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# S3 client with anonymous (unsigned) access
s3 = boto3.client(
    's3',
    config=Config(signature_version=UNSIGNED)
)

def fetch_html_from_warc_range(
    warc_key: str,
    offset: int,
    length: int,
    target_url: str
) -> Optional[str]:
    """
    Fetch byte range from WARC file in S3 and extract HTML for the target URL.
    """
    try:
        logger.info(f"Fetching range bytes={offset}-{offset+length-1} for {target_url}")
        obj = s3.get_object(
            Bucket='commoncrawl',
            Key=warc_key,
            Range=f'bytes={offset}-{offset + length - 1}'
        )

        warc_stream = obj['Body']
        for record in ArchiveIterator(warc_stream):
            if (record.rec_type == 'response' and
                record.rec_headers.get_header('WARC-Target-URI') == target_url):
                html_bytes = record.content_stream().read()
                try:
                    return html_bytes.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    logger.warning(f"Decode error for {target_url}, using latin-1 fallback")
                    return html_bytes.decode('latin-1', errors='ignore')

        logger.warning(f"No matching WARC record found for {target_url}")
        return None

    except Exception as e:
        logger.error(f"Fetch failed for {target_url}: {type(e).__name__} - {e}")
        return None


def main(
    input_csv: str = "data/raw/vi_aligned.csv",
    output_dir: str = "data/raw/vi_html",
    limit: int = None,
    overwrite: bool = False
):
    """
    Main execution: read aligned metadata, fetch HTML, save to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading aligned metadata from {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logger.error(f"Input CSV not found: {input_csv}")
        sys.exit(1)

    if limit is not None:
        df = df.head(limit)
        logger.info(f"Limiting to first {limit} records")

    successful = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching HTML"):
        url = row['url']
        warc_key = row['warc_filename']
        offset = int(row['offset'])
        length = int(row['length'])

        # Sanitize filename (avoid invalid characters)
        safe_url = url.replace('/', '_').replace(':', '_').replace('?', '_')
        output_file = output_path / f"{safe_url}.html"

        if output_file.exists() and not overwrite:
            logger.info(f"Skipping existing file: {output_file}")
            successful += 1
            continue

        html = fetch_html_from_warc_range(warc_key, offset, length, url)
        if html:
            output_file.write_text(html, encoding='utf-8', errors='ignore')
            successful += 1
            logger.info(f"Saved HTML: {output_file}")
        else:
            failed += 1

    logger.info(f"Completed: {successful} successful, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch HTML from Common Crawl WARC ranges.")
    parser.add_argument("--input_csv", default="data/raw/vi_aligned.csv",
                        help="Path to aligned metadata CSV")
    parser.add_argument("--output_dir", default="data/raw/vi_html",
                        help="Directory to save extracted HTML files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of URLs to process")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing HTML files")

    args = parser.parse_args()

    main(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite
    )
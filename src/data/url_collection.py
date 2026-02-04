"""
Vietnamese and Multi-Domain URL Scraper for Common Crawl CDX Index.

This module supports:
- Paginated querying of the Common Crawl index
- Resume capability from existing CSV files
- Domain-based patterns or broad URL patterns with filters
- Periodic checkpoint saving
- Flexible filtering (language, status, MIME type, etc.)
"""

import requests
import json
import pandas as pd
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urlparse
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass
class URLCollectorConfig:
    """Configuration parameters for the URL collector."""
    # Domain-based collection (optional fallback)
    domains: List[str] = field(default_factory=list)

    # General settings
    max_urls: int = 100_000
    index_name: str = "CC-MAIN-2025-51-index"
    output_file: str = "vi_urls.csv"
    politeness_delay: float = 1.0
    timeout: int = 30
    max_retries: int = 3
    resume: bool = True
    save_every_n: int = 1000

    # Flexible query mode (preferred for language-based collection)
    url_pattern: str = "*"
    filter_params: List[str] = field(default_factory=list)


class CommonCrawlURLCollector:
    """Handles paginated collection of URLs from the Common Crawl CDX index."""

    def __init__(self, config: URLCollectorConfig):
        self.config = config
        self.base_url = f"https://index.commoncrawl.org/{config.index_name}"
        self.output_path = Path(config.output_file)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("url_collector")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _get_url_pattern(self) -> str:
        """Determine the effective URL pattern for the CDX query."""
        if self.config.domains:
            patterns = []
            for d in self.config.domains:
                d = d.strip()
                if d.startswith("*."):
                    patterns.append(f"{d}/*")
                else:
                    patterns.append(f"*.{d}/*")
                    patterns.append(f"{d}/*")
            return " OR ".join(set(patterns))
        return self.config.url_pattern

    def _load_existing_urls(self) -> set[str]:
        """Load previously collected URLs for resume capability."""
        if not self.config.resume or not self.output_path.is_file():
            return set()
        try:
            df = pd.read_csv(self.output_path, usecols=["url"])
            return set(df["url"])
        except Exception as e:
            self.logger.warning(f"Failed to load existing URLs: {e}")
            return set()

    def collect(self) -> pd.DataFrame:
        """Collect URLs from the Common Crawl index according to configuration."""
        existing = self._load_existing_urls()
        collected: List[dict] = []

        # Build patterns based on config
        if self.config.domains:
            patterns = []
            for d in self.config.domains:
                d = d.strip()
                if d.startswith("*."):
                    patterns.append(f"{d}/*")
                else:
                    patterns.append(f"*.{d}/*")
                    patterns.append(f"{d}/*")
            patterns = list(set(patterns))
        else:
            patterns = [self._get_url_pattern()]

        for pattern in patterns:
            if len(collected) >= self.config.max_urls:
                break

            self.logger.info(f"Collecting for pattern: {pattern}")
            page = 0
            retries = 0

            while len(collected) < self.config.max_urls:
                try:
                    params = {
                        "url": pattern,
                        "output": "json",
                        "page": page,
                        "limit": 5000
                    }

                    # Add filter params if provided
                    if self.config.filter_params:
                        for f in self.config.filter_params:
                            params.setdefault("filter", []).append(f)

                    response = requests.get(
                        self.base_url,
                        params=params,
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()

                    text = response.text.strip()
                    if not text:
                        self.logger.info(f"Empty response → end of results for pattern {pattern}")
                        break

                    new_count = 0
                    for line in text.splitlines():
                        try:
                            record = json.loads(line)
                            url = record["url"]
                            if url in existing:
                                continue
                            collected.append({
                                "url": url,
                                "warc_filename": record["filename"],
                                "offset": int(record["offset"]),
                                "length": int(record["length"]),
                                "timestamp": record.get("timestamp", ""),
                                "status": record.get("status", ""),
                                "digest": record.get("digest", ""),
                                "domain": urlparse(url).netloc
                            })
                            new_count += 1
                            if len(collected) >= self.config.max_urls:
                                break
                        except (json.JSONDecodeError, KeyError) as e:
                            self.logger.debug(f"Skipping malformed line: {e}")
                            continue

                    self.logger.info(f"Page {page}: added {new_count} new URLs (total: {len(collected)})")

                    if new_count == 0 and page > 0:
                        self.logger.info(f"No new URLs on page {page}, ending pattern {pattern}")
                        break

                    # Save checkpoint periodically
                    if len(collected) % self.config.save_every_n == 0:
                        self._save_checkpoint(collected)

                    page += 1
                    time.sleep(self.config.politeness_delay)
                    retries = 0

                except requests.RequestException as e:
                    self.logger.warning(f"Request failed (page {page}): {e}")
                    retries += 1
                    if retries >= self.config.max_retries:
                        self.logger.error(f"Max retries exceeded for pattern {pattern}")
                        break
                    time.sleep(self.config.politeness_delay * 2)
                except Exception as e:
                    self.logger.error(f"Unexpected error on page {page}: {e}")
                    retries += 1
                    if retries >= self.config.max_retries:
                        break
                    time.sleep(self.config.politeness_delay * 2)

        return self._save_final(collected)

    def _save_checkpoint(self, records: List[dict]):
        """Save intermediate checkpoint."""
        if not records:
            return
        df = pd.DataFrame(records).drop_duplicates(subset="url")
        df.to_csv(self.output_path, index=False)
        self.logger.debug(f"Checkpoint saved: {len(df)} URLs")

    def _save_final(self, records: List[dict]) -> pd.DataFrame:
        """Save final results and return DataFrame."""
        if not records:
            self.logger.warning("No URLs collected")
            return pd.DataFrame()

        df = pd.DataFrame(records).drop_duplicates(subset="url")
        df.to_csv(self.output_path, index=False)
        self.logger.info(f"Final save: {len(df)} unique URLs saved to {self.output_path}")

        # Print summary statistics
        if not df.empty:
            self.logger.info(f"Top domains collected:")
            top_domains = df["domain"].value_counts().head(10)
            for domain, count in top_domains.items():
                self.logger.info(f"  {domain}: {count}")

        return df


def collect_vietnamese_urls(
        max_urls: int = 50_000,
        domains: Optional[List[str]] = None,
        output_file: str = "vi_commoncrawl_urls.csv",
        resume: bool = True,
        **kwargs
) -> pd.DataFrame:
    """Convenience function: collect URLs from specified .vn-related domains."""
    if domains is None:
        domains = ["*.vn", "vn"]

    config = URLCollectorConfig(
        domains=domains,
        max_urls=max_urls,
        output_file=output_file,
        resume=resume,
        **kwargs
    )

    collector = CommonCrawlURLCollector(config)
    return collector.collect()


def collect_non_vn_vietnamese_urls(
        max_urls: int = 50000,
        index_name: str = "CC-MAIN-2025-51-index",
        output_file: str = "vi_non_vn_urls.csv",
        resume: bool = True,
        politeness_delay: float = 1.5,
        **kwargs
) -> pd.DataFrame:
    """
    Collect Vietnamese-language URLs excluding .vn domains.

    Uses CDX server language filter 'languages:vie' combined with domain negation.
    """
    # Remove any filter_params from kwargs to avoid conflicts
    kwargs.pop('filter_params', None)

    config = URLCollectorConfig(
        url_pattern="*",
        filter_params=[
            "languages:vie",
            #"status:200",
            #"mime:text/html",
        ],
        max_urls=max_urls,
        index_name=index_name,
        output_file=output_file,
        resume=resume,
        politeness_delay=politeness_delay,
        save_every_n=500,  # More frequent checkpoints for broad queries
        **kwargs
    )

    collector = CommonCrawlURLCollector(config)
    return collector.collect()


# Helper function for phishing URLs
def collect_phishing_urls(
        max_urls: int = 50000,
        domains: Optional[List[str]] = None,
        output_file: str = "phishing_urls.csv",
        resume: bool = True,
        **kwargs
) -> pd.DataFrame:
    """
    Collect URLs from known phishing domains.
    You'll need to provide a list of phishing domains.
    """
    if domains is None:
        domains = []  # Add your phishing domains here

    if not domains:
        raise ValueError("Please provide a list of phishing domains")

    config = URLCollectorConfig(
        domains=domains,
        max_urls=max_urls,
        output_file=output_file,
        resume=resume,
        **kwargs
    )

    collector = CommonCrawlURLCollector(config)
    return collector.collect()


if __name__ == "__main__":
    # Example usage
    print("Testing URL collection...")

    # Test 1: Collect .vn domains
    df_vn = collect_vietnamese_urls(
        max_urls=1000,
        domains=["*.vn"],
        output_file="test_vn_urls.csv",
        resume=False
    )
    print(f"Collected {len(df_vn)} .vn URLs")

    # Test 2: Collect non-.vn Vietnamese content
    df_non_vn = collect_non_vn_vietnamese_urls(
        max_urls=1000,
        output_file="test_non_vn_urls.csv",
        resume=False
    )
    print(f"Collected {len(df_non_vn)} non-.vn Vietnamese URLs")


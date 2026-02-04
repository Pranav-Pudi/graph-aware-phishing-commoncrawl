"""
Common Crawl WARC → HTML → Clean Text Pipeline
Features: resume support, batch processing, progress tracking, quality filtering
"""

import pandas as pd
import requests
import io
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from tqdm import tqdm
import hashlib

@dataclass
class PipelineConfig:
    url_csv: str = "vi_commoncrawl_urls.csv"
    html_dir: str = "data/html"
    text_dir: str = "data/text"
    max_threads: int = 16
    batch_size: int = 800
    max_retries: int = 3
    request_timeout: int = 60
    politeness_delay: float = 0.4
    min_text_length: int = 120
    resume: bool = True
    skip_diagnostics: bool = True

class WARCPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.html_root = Path(config.html_dir)
        self.text_root = Path(config.text_dir)
        self.html_root.mkdir(parents=True, exist_ok=True)
        self.text_root.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.html_root / ".pipeline_progress.json"
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("warc_pipeline")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    def _url_to_stem(self, url: str) -> str:
        digest = hashlib.sha256(url.encode()).hexdigest()[:14]
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in url[:48])
        return f"{safe}_{digest}"

    def _load_progress(self) -> Dict[str, List[str]]:
        if not self.progress_file.is_file():
            return {"processed": [], "failed": []}
        try:
            with self.progress_file.open() as f:
                return json.load(f)
        except Exception:
            self.logger.warning("Invalid progress file → starting fresh")
            return {"processed": [], "failed": []}

    def _save_progress(self, processed: List[str], failed: List[str]):
        data = {
            "processed": processed,
            "failed": failed,
            "last_updated": time.time()
        }
        with self.progress_file.open("w") as f:
            json.dump(data, f, indent=2)

    def _get_remaining_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.url_csv)
        if not self.config.resume:
            return df

        progress = self._load_progress()
        processed = set(progress["processed"])
        failed = set(progress["failed"])

        def keep_row(r):
            u = r["url"]
            if u in processed or u in failed:
                return False
            stem = self._url_to_stem(u)
            return not (self.html_root / f"{stem}.html").exists() or \
                   not (self.text_root / f"{stem}.txt").exists()

        remaining = df[df.apply(keep_row, axis=1)]
        self.logger.info(f"Total URLs: {len(df):,} | Already done/failed: {len(df)-len(remaining):,} | Remaining: {len(remaining):,}")
        return remaining

    def fetch_warc_record(self, row: pd.Series) -> Optional[Path]:
        url = row["url"]
        warc = row["warc_filename"]
        off = int(row["offset"])
        length = int(row["length"])

        if self.config.skip_diagnostics and "crawldiagnostics" in warc.lower():
            return None

        target = self.html_root / f"{self._url_to_stem(url)}.html"
        if target.exists():
            return target

        http_url = f"https://data.commoncrawl.org/{warc}"
        headers = {"Range": f"bytes={off}-{off + length - 1}"}

        for attempt in range(self.config.max_retries):
            try:
                r = requests.get(http_url, headers=headers, timeout=self.config.request_timeout)
                r.raise_for_status()

                for record in ArchiveIterator(io.BytesIO(r.content)):
                    if record.rec_type == "response":
                        html = record.content_stream().read().decode("utf-8", errors="ignore")
                        target.write_text(html, encoding="utf-8")
                        time.sleep(self.config.politeness_delay)
                        return target
            except Exception as e:
                self.logger.warning(f"[{attempt+1}/{self.config.max_retries}] {url} → {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(1.5 ** attempt)

        return None

    def extract_clean_text(self, html_path: Path) -> Optional[Path]:
        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")

            for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean = "\n".join(lines)

            if len(clean) < self.config.min_text_length:
                return None

            txt_path = self.text_root / f"{html_path.stem}.txt"
            txt_path.write_text(clean, encoding="utf-8")
            return txt_path
        except Exception as e:
            self.logger.warning(f"Extraction failed {html_path.name}: {e}")
            return None

    def process_batch(self, batch: pd.DataFrame) -> Tuple[List[str], List[str]]:
        processed, failed = [], []

        # Download HTML
        html_paths = []
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as ex:
            future_to_url = {
                ex.submit(self.fetch_warc_record, row): row["url"]
                for _, row in batch.iterrows()
            }
            for future in tqdm(as_completed(future_to_url), total=len(future_to_url), desc="HTML"):
                url = future_to_url[future]
                try:
                    path = future.result()
                    if path:
                        html_paths.append(path)
                        processed.append(url)
                    else:
                        failed.append(url)
                except Exception as e:
                    self.logger.error(f"Download error {url}: {e}")
                    failed.append(url)

        # Extract text
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as ex:
            future_to_html = {ex.submit(self.extract_clean_text, p): p for p in html_paths}
            for future in tqdm(as_completed(future_to_html), total=len(future_to_html), desc="Text "):
                try:
                    future.result()  # we don't need the path, just success
                except Exception:
                    pass  # already logged

        return processed, failed

    def run(self) -> Dict:
        df = self._get_remaining_df()
        if df.empty:
            self.logger.info("No remaining work.")
            return {"total": 0, "processed": 0, "failed": 0}

        progress = self._load_progress()
        all_processed = progress["processed"].copy()
        all_failed = progress["failed"].copy()

        n_batches = (len(df) + self.config.batch_size - 1) // self.config.batch_size

        for i in range(n_batches):
            start = i * self.config.batch_size
            end = min(start + self.config.batch_size, len(df))
            batch = df.iloc[start:end]

            self.logger.info(f"Batch {i+1}/{n_batches} — {len(batch)} URLs")
            proc, fail = self.process_batch(batch)

            all_processed.extend(proc)
            all_failed.extend(fail)
            self._save_progress(all_processed, all_failed)

        stats = {
            "total": len(df),
            "processed": len(all_processed),
            "failed": len(all_failed),
            "success_pct": len(all_processed) / len(df) * 100 if df.shape[0] > 0 else 0
        }

        self.logger.info(
            f"Pipeline finished | Processed: {stats['processed']:,} | "
            f"Failed: {stats['failed']:,} | Success: {stats['success_pct']:.1f}%"
        )
        return stats


def run_warc_pipeline(
    url_csv: str = "vi_commoncrawl_urls.csv",
    html_dir: str = "data/html",
    text_dir: str = "data/text",
    **kwargs
) -> Dict:
    config = PipelineConfig(
        url_csv=url_csv,
        html_dir=html_dir,
        text_dir=text_dir,
        **kwargs
    )
    pipeline = WARCPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    run_warc_pipeline(
        url_csv="data/vietnamese_urls.csv",
        html_dir="data/html_vi",
        text_dir="data/text_vi",
        max_threads=12,
        batch_size=600
    )

import requests
import json
from typing import Optional, Dict, List
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

LATEST_CRAWL = "CC-MAIN-2025-51"
FALLBACK_CRAWL = "CC-MAIN-2025-50"


def create_session():
    """Factory to create a fresh, isolated Session per process."""
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'NUS-Student-Phishing-Project',
        'Accept': 'application/json',
        'Accept-Encoding': 'identity'
    })
    return s


def query_cc_index(
        url: str,
        crawl: str = LATEST_CRAWL,
        max_retries: int = 5,
        backoff_factor: float = 2.5,
        verbose: bool = False
) -> Optional[Dict]:
    """
    Process-safe query: creates its own Session.
    """
    session = create_session()  # Fresh session per call / process
    api_url = f"https://index.commoncrawl.org/{crawl}-index?url={url}&output=json"

    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"[PID {mp.current_process().pid}] Querying {url} (attempt {attempt + 1})")

            response = session.get(api_url, timeout=20)
            response.raise_for_status()

            text = response.text.strip()
            if not text:
                return None

            lines = text.split('\n')
            captures = []
            for line in lines:
                if line.strip():
                    try:
                        captures.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if not captures:
                return None

            latest = max(captures, key=lambda x: x.get('timestamp', '00000000000000'))
            return {
                'url': url,
                'timestamp': latest.get('timestamp'),
                'warc_filename': latest.get('filename'),
                'offset': latest.get('offset'),
                'length': latest.get('length'),
                'status': latest.get('status'),
                'mime': latest.get('mime')
            }

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"[PID {mp.current_process().pid}] Attempt {attempt + 1} failed: {type(e).__name__} - {e}")
            if attempt == max_retries - 1:
                if crawl != FALLBACK_CRAWL:
                    return query_cc_index(url, FALLBACK_CRAWL, max_retries=3, verbose=verbose)
                return None
            time.sleep(backoff_factor ** attempt)

    return None


def batch_query_cc_index(
        urls: List[str],
        crawl: str = LATEST_CRAWL,
        max_workers: int = None,
        show_progress: bool = True,
        verbose: bool = False
) -> List[Dict]:
    """
    Parallel batch using ProcessPoolExecutor – each process gets its own Session.
    """
    if max_workers is None:
        max_workers = mp.cpu_count()

    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(query_cc_index, url, crawl, verbose=verbose): url
            for url in urls
        }

        iterator = tqdm(as_completed(future_to_url), total=len(urls), desc="Aligning", disable=not show_progress)
        for future in iterator:
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed.append(url)
            except Exception as e:
                print(f"Process error for {url}: {e}")
                failed.append(url)

    if failed:
        print(f"Failed {len(failed)} / {len(urls)} URLs. First few: {failed[:5]}")

    return results
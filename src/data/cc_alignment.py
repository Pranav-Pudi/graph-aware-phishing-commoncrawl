import requests
import json
from typing import Optional, Dict, List

LATEST_CRAWL = "CC-MAIN-2025-51"  # Latest index as of January 2026

def query_cc_index(url: str, crawl: str = LATEST_CRAWL) -> Optional[Dict]:
    """
    Query Common Crawl Index for a URL. Returns most recent capture metadata.
    """
    api_url = f"https://index.commoncrawl.org/{crawl}-index?url={url}&output=json"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        if not lines:
            return None
        captures = [json.loads(line) for line in lines]
        latest = max(captures, key=lambda x: x.get('timestamp', ''))
        return {
            'url': url,
            'timestamp': latest.get('timestamp'),
            'warc_filename': latest.get('filename'),
            'offset': latest.get('offset'),
            'length': latest.get('length'),
            'status': latest.get('status'),
            'mime': latest.get('mime')
        }
    except Exception as e:
        print(f"Error querying {url}: {e}")
        return None

def query_vietnamese_samples(limit: int = 100, crawl: str = LATEST_CRAWL) -> List[str]:
    """
    Sample Vietnamese pages via lang:vie + .vn TLD filter.
    Note: Index API does not support direct random sampling; this uses broad wildcard.
    """
    # Example broad Vietnamese query (adjust for more diversity)
    sample_query = f"https://index.commoncrawl.org/{crawl}-index?url=*.vn/*&filter=lang:vie&output=json&page=0"
    try:
        response = requests.get(sample_query, timeout=20)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        urls = []
        for line in lines[:limit]:
            if line.strip():
                data = json.loads(line)
                urls.append(data.get('url'))
        return urls
    except Exception as e:
        print(f"Error sampling Vietnamese: {e}")
        return []
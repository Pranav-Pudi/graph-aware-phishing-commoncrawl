import argparse
import logging
import time
import pandas as pd
import whois
from tqdm import tqdm
from pathlib import Path

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def lookup_domain(domain):
    """Perform WHOIS lookup with error handling."""
    domain = domain.strip().lower()
    # Basic validation: must have a dot and be reasonable length
    if not domain or '.' not in domain or len(domain) < 4:
        return {'domain': domain, 'status': 'invalid'}
    
    try:
        w = whois.whois(domain)
        return {
            'domain': domain,
            'registrar': w.registrar,
            'creation_date': str(w.creation_date) if w.creation_date else None,
            'expiration_date': str(w.expiration_date) if w.expiration_date else None,
            'name_servers': str(w.name_servers) if w.name_servers else None,
            'registrant_org': w.registrant_org,
            'status': str(w.status) if w.status else "active"
        }
    except Exception as e:
        return {'domain': domain, 'status': f'error: {str(e)[:50]}'}

def main():
    parser = argparse.ArgumentParser(description="Bulk WHOIS lookup for Network 99")
    parser.add_argument("--input", type=str, default="suspicious_network99_domains.txt")
    parser.add_argument("--output", type=str, default="suspicious_network99_whois.csv")
    args = parser.parse_args()

    # 1. Load Domains with robust reading
    domains = []
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file NOT FOUND at: {input_path.absolute()}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip().replace('"', '').replace("'", "").replace(",", "")
            if clean_line and '.' in clean_line:
                domains.append(clean_line)

    logger.info(f"Loaded {len(domains)} unique domains for analysis.")

    if not domains:
        logger.error("No valid domains found in the text file. Check the content!")
        return

    # 2. Run Lookups with Rate Limiting
    results = []
    logger.info("Starting WHOIS lookups (1s delay per domain to avoid IP ban)...")
    
    for domain in tqdm(domains, desc="WHOIS Progress"):
        res = lookup_domain(domain)
        results.append(res)
        time.sleep(1.0)  # CRITICAL: Prevents Atlas from being blocked

    # 3. Save Results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"WHOIS results saved to {args.output}")

    # 4. Summary Statistics (Safe check if columns exist)
    if 'registrar' in df.columns:
        print("\n--- TOP REGISTRARS ---")
        print(df['registrar'].value_counts().head(10))
    
    if 'status' in df.columns:
        print("\n--- STATUS SUMMARY ---")
        print(df['status'].value_counts().head(5))

if __name__ == "__main__":
    main()


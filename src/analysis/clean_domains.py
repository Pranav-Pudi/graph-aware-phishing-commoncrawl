#!/usr/bin/env python3
import argparse
import pandas as pd
import re

def extract_domain(url_id: str) -> str | None:
    match = re.search(r'___([^_]+)', str(url_id))
    if not match:
        return None
    domain_part = match.group(1)
    parts = url_id.split('___')[-1].split('_')
    if len(parts) >= 2:
        suffix = parts[1]
        if suffix in ('vn', 'com', 'net', 'org', 'sg'):
            return f"{domain_part}.{suffix}"
    return domain_part

def main():
    parser = argparse.ArgumentParser(description="Extract domains from Network 99")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to gnn_solver_results.parquet")
    parser.add_argument("--output", type=str, default="suspicious_network99_domains.txt",
                        help="Output file for domains")
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_parquet(args.input)
    net99 = df[df['network_id'] == 99]

    raw_ids = net99['id'].unique()
    domains = set()

    for rid in raw_ids:
        domain = extract_domain(rid)
        if domain:
            domains.add(domain)

    with open(args.output, 'w') as f:
        for d in sorted(list(domains)):
            f.write(f"{d}\n")

    print(f"Saved {len(domains)} domains to {args.output}")

if __name__ == "__main__":
    main()




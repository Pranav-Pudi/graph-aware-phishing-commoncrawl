#!/usr/bin/env python3
"""
Inspect and summarize BERTopic model topics.

Loads a saved BERTopic model, extracts topic information (counts, names, keywords),
prints top topics and detailed keywords for the largest topic, and saves a CSV topic dictionary.
"""

import argparse
import logging
import os
from pathlib import Path

from bertopic import BERTopic

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Inspect and summarize BERTopic model topics")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the saved BERTopic model (e.g., outputs/bertopic_model)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/topic_dictionary.csv",
        help="Path to save the topic info CSV (default: outputs/topic_dictionary.csv)"
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_path = Path(args.output)

    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        return

    logger.info(f"Loading BERTopic model from: {model_path}")
    try:
        topic_model = BERTopic.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("Extracting topic information...")
    topic_info = topic_model.get_topic_info()

    # Print top 20 topics
    print("\n--- TOP 20 SEMANTIC CLUSTERS ---")
    print(topic_info[['Topic', 'Count', 'Name']].head(20).to_string(index=False))

    # Deep dive into the largest topic (usually Topic 0 or -1)
    if not topic_info.empty:
        largest_topic = topic_info.iloc[0]['Topic']
        print(f"\n--- DETAILED KEYWORDS FOR TOPIC {largest_topic} ---")
        keywords = topic_model.get_topic(largest_topic)
        if keywords:
            print([word for word, score in keywords])
        else:
            print("(No representative keywords available for this topic)")
    else:
        print("\nNo topics found in the model.")

    # Save full topic info
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topic_info.to_csv(output_path, index=False)
    logger.info(f"Topic dictionary saved to: {output_path}")


if __name__ == "__main__":
    main()


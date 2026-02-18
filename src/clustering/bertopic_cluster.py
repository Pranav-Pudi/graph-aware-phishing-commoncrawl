import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Logging setup - flushes frequently for PBS visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("outputs/bertopic_run.log")]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/semantic_results.parquet")
    parser.add_argument("--model-dir", type=str, default="outputs/bertopic_model")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Reading: {args.input}")
    df = pd.read_parquet(args.input)
    docs = df['text'].astype(str).replace('', '[EMPTY]').tolist()
    logger.info(f"Processing {len(docs)} documents")

    # 2. Embedding Model
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 3. MULTI-PROCESS EMBEDDING (The Speed Hack)
    logger.info("Starting Multi-Process Encoding on all available cores...")
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(docs, pool, batch_size=args.batch_size)
    model.stop_multi_process_pool(pool)
    logger.info(f"Embeddings finished. Shape: {embeddings.shape}")

    # 4. Dimensionality Reduction (UMAP)
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        low_memory=True,
        random_state=42
    )

    # 5. Clustering (HDBSCAN)
    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # 6. BERTopic
    # We pass 'None' to embedding_model because we provide pre-computed embeddings
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
        calculate_probabilities=False
    )

    # 7. Fit & Transform using the pre-computed embeddings
    logger.info("Running BERTopic clustering (Clustering only)...")
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    # 8. Save Results
    df['topic'] = topics
    df.to_parquet(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # 9. Save Model
    topic_model.save(args.model_dir, serialization="safetensors")
    logger.info(f"Model saved to {args.model_dir}")


if __name__ == "__main__":
    main()
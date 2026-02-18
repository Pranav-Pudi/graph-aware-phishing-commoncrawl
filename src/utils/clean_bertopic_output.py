import pandas as pd
df = pd.read_parquet('/hpctmp/e1430653/project/outputs/semantic_results.parquet')

# 1. Remove the "Empty" Topic 0
# 2. Keep Topic -1 (Noise) as it might contain unique content nodes
# 3. Filter out technical errors (Topics 1, 7, 9)
junk_topics = [0, 1, 7, 9]
clean_df = df[~df['topic'].isin(junk_topics)]

print(f"Final Usable Rows: {len(clean_df)}")
clean_df.to_parquet('/hpctmp/e1430653/project/outputs/final_gnn_features.parquet')

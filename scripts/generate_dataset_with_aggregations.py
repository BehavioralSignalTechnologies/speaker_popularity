import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
sys.path.append("..")
from audio_features.aggregate import run_aggregation, run_embedding_aggregation

merged_metadata_df = pd.read_csv("../metadata/merged_metadata_popularity.csv")

def get_related_result(row):
    for entry in os.listdir('../modeling_api_results_embeddings'):
        if not entry.endswith(".wav.json"):
            continue

        if entry.split(".wav.json")[0] == row['filename'].split(".wav")[0]:
            with open(os.path.join('../modeling_api_results_embeddings', entry)) as f:
                return json.load(f)

    print(f"File {row['url']} not found in modeling_api_results")
    return None


features_metadata = []

for idx, row in merged_metadata_df.iterrows():
    # Modeling api result
    result = get_related_result(row)

    if result is None:
        continue

    # Aggregate with Thodoris script
    try:
        mean_posteriors = run_aggregation(result)
        mean_embeddings = run_embedding_aggregation(result)
    except Exception as e:
        print(result, "will not be included!")
        continue

    # Flatten features
    mean_posteriors_flattened = {}
    for task in mean_posteriors:
        for cls in mean_posteriors[task]:
            if type(mean_posteriors[task][cls]) == dict:
                for metric, value in mean_posteriors[task][cls].items():
                    mean_posteriors_flattened[f"{task}_{cls}_{metric}"] = value
            else:
                mean_posteriors_flattened[f"{task}_{cls}"] = mean_posteriors[task][cls]

    mean_posteriors_flattened['features_embedding_mean'] = mean_embeddings['mean'].tolist()
    mean_posteriors_flattened['features_embedding_std'] = mean_embeddings['std'].tolist()
    entry = {idx: value for idx, value in row.items()}
    entry.update(mean_posteriors_flattened)
    features_metadata.append(entry)

features_metadata_df = pd.DataFrame(features_metadata)
print(f"Length of generated dataset: {len(features_metadata_df)}")
features_metadata_df.to_csv("../metadata/merged_metadata_popularity_features.csv")
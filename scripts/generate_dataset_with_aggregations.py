import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json

import tqdm

sys.path.append("..")
from audio_features.aggregate import run_aggregation, run_embedding_aggregation

merged_metadata_df = pd.read_csv("../metadata/merged_metadata_popularity_std.csv")


def get_acoustic_features(result):
    filename = result['filename']
    res = np.load(f"../mfccs/{filename}.npz")
    return res['arr_0'], res['arr_1']

def get_related_result(row):
    entry = row['filename']+".json"
    try:
        with open(os.path.join('../modeling_api_results_embeddings', entry)) as f:
            return json.load(f)
    except Exception as e:
        print(e)
    print(f"File {row['url']} not found in modeling_api_results")
    return None


features_metadata = []

for idx, row in tqdm.tqdm(merged_metadata_df.iterrows(), total=len(merged_metadata_df)):
    # Modeling api result
    result = get_related_result(row)

    if result is None:
        continue

    # Aggregate with Thodoris script
    try:
        mean_posteriors = run_aggregation(result)
        mean_embeddings = run_embedding_aggregation(result)
        mfccs_mean, mfccs_std = get_acoustic_features(row)
    except Exception as e:
        print(row, "will not be included!")
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
    mean_posteriors_flattened['mfccs_mean'] = mfccs_mean.tolist()
    mean_posteriors_flattened['mfccs_std'] = mfccs_std.tolist()
    entry = {idx: value for idx, value in row.items()}
    entry.update(mean_posteriors_flattened)
    features_metadata.append(entry)

features_metadata_df = pd.DataFrame(features_metadata)
print(f"Length of generated dataset: {len(features_metadata_df)}")
features_metadata_df.to_csv("../metadata/merged_metadata_popularity_features_std.csv")
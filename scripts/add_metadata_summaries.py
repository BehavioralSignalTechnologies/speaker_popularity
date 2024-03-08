import os
import json
import datetime
import numpy as np
import pandas as pd

tasks = ["emotion", "positivity", "strength"]

if __name__ == '__main__':
    df = pd.read_csv("../metadata/merged_metadata_popularity.csv")
    for idx, row in df.iterrows():
        filename = row["filename"]
        json_summary_file = os.path.join("../modeling_api_results_embeddings", filename + "_summary.json")
        if not os.path.exists(json_summary_file):
            continue
        with open(json_summary_file, 'r') as f:
            json_summary = json.load(f)
        try:
            df.at[idx, "speaker_turns"] = json_summary["diarization"]["SPEAKER_00"]
        except KeyError:
            continue
        df.at[idx, "gender"] = max(json_summary["gender"], key=json_summary["gender"].get)
        df.at[idx, "age"] = max(json_summary["age"], key=json_summary["age"].get)
        for t in tasks:
            task_dict = json_summary[t]
            label_total = sum(task_dict.values())
            for key, value in task_dict.items():
                task_label = f"{t}_{key}"
                task_value = round(value / label_total, 2)
                df.at[idx, task_label] = task_value
    df.to_csv("../metadata/merged_metadata_popularity_summaries.csv", index=False)
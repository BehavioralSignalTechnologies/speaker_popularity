import datetime
from functools import partial

import numpy as np
import pandas as pd

def map_to_label(value, ranges, labels):
    if len(ranges) != len(labels):
        raise ValueError("Number of labels must be equal to the number of ranges")

    for range, label in zip(ranges, labels):
        if value >= range[0] and value <= range[1]:
            return label

    raise ValueError(f"value {value} not located between ranges: {ranges}")


views_mapping_5 = partial(map_to_label,
                          ranges=[[-np.inf, -1.5], [-1.5, -0.5], [-0.5, 0.5], [0.5, 1.5], [1.5, np.inf]],
                          labels=["very_low", "low", "medium", "high", "very_high"])

views_mapping_3 = partial(map_to_label,
                          ranges=[[-np.inf, -0.5], [-0.5, 0.5], [0.5, np.inf]],
                          labels=["low", "medium", "high"])

if __name__ == '__main__':
    df = pd.read_csv("../metadata/merged_metadata.csv")
    df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)))
    df['published_date'] = df['published_date'].apply(
        lambda x: datetime.datetime.fromtimestamp(int(x)))
    df['film_year'] = df['film_date'].apply(lambda it: it.year)
    df['published_year'] = df['published_date'].apply(lambda it: it.year)

    # Filter dataset
    print(
        f"Total talks between 2010-2016: {len(df.loc[(df['film_year'] >= 2010) & (df['film_year'] <= 2016), :])}")

    # We will use videos between 2010-2016
    df = df.loc[(df['film_year'] >= 2010) & (df['film_year'] <= 2016), :]
    # Num_speakers == 1
    df = df.loc[df['num_speaker'] == 1, :]

    # Generate views target set
    log_views = np.log(df['views'])
    df['log_views_norm'] = (log_views - log_views.mean()) / log_views.std()
    df['log_views_norm_cat'] = df['log_views_norm'].apply(views_mapping_3)

    # Generate comments target set
    log_comments = np.log(df['comments'])
    df['log_comments_norm'] = (log_comments - log_comments.mean()) / log_comments.std()
    df['log_comments_norm_cat'] = df['log_comments_norm'].apply(views_mapping_3)

    # Generate comments/views set
    comments_per_view = df['comments'] / df['views']
    log_comments_per_view = np.log(comments_per_view)
    df['log_comments_per_view_norm'] = (log_comments_per_view - log_comments_per_view.mean()) / log_comments_per_view.std()
    df['log_comments_norm_cat'] = df['log_comments_per_view_norm'].apply(views_mapping_3)

    df.to_csv("../metadata/merged_metadata_popularity.csv", index=False)
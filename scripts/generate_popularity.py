import datetime
import numpy as np
import pandas as pd

def map_to_label5(value):
    if value < -1.5:
        return "very_low"
    elif value < -0.5:
        return "low"
    elif value < 0.5:
        return "medium"
    elif value < 1.5:
        return "high"
    else:
        return "very_high"

def map_to_label3(value):
    if value < -0.5:
        return "low"
    elif value < 0.5:
        return "normal"
    else:
        return "high"

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

    # Generate target set
    log_views = np.log(df['views'])
    df['log_views_norm'] = (log_views - log_views.mean()) / log_views.std()
    df['log_views_norm_cat'] = df['log_views_norm'].apply(map_to_label3)

    df['log_views_norm_cat'] = df['log_views_norm'].apply(map_to_label3)

    df.to_csv("metadata/merged_metadata_popularity.csv", index=False)
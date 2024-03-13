import ast
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


views_mapping_3_std = partial(map_to_label,
                          ranges=[[-np.inf, -1], [-1, 1], [1, np.inf]],
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

    # Replace nan occupation with ""
    df['speaker_occupation']: pd.Series = df['speaker_occupation'].fillna("")

    # Generate views target set
    log_views = np.log(df['views'])
    views = df['views']
    df['log_views_norm'] = (log_views - log_views.mean()) / log_views.std()
    df['views_norm'] = (views - views.mean()) / views.std()
    df['log_views_norm_cat'] = df['log_views_norm'].apply(views_mapping_3_std)

    # Generate comments target set
    log_comments = np.log(df['comments'])
    comments = df['comments']
    df['log_comments_norm'] = (log_comments - log_comments.mean()) / log_comments.std()
    df['comments_norm'] = (comments - comments.mean()) / comments.std()
    df['log_comments_norm_cat'] = df['log_comments_norm'].apply(views_mapping_3_std)

    # Generate comments/views set
    comments_per_view = df['comments'] / df['views']
    df['comments_per_view_norm'] = (comments_per_view - comments_per_view.mean()) / comments_per_view.std()

    log_comments_per_view = np.log(comments_per_view)
    df['log_comments_per_view_norm'] = (log_comments_per_view - log_comments_per_view.mean()) / log_comments_per_view.std()
    df['log_comments_per_view_norm_cat'] = df['log_comments_per_view_norm'].apply(views_mapping_3_std)
    df["ratings_ast"] = df['ratings'].apply(ast.literal_eval)
    df['ratings_sum'] = df["ratings_ast"].apply(lambda x: np.sum([tag['count'] for tag in x]))
    ratings_views =  np.log(df['ratings_sum'] / df['views'])
    df['log_ratings_views_norm'] = (ratings_views - ratings_views.mean()) / ratings_views.std()
    df['log_ratings_views_norm_cat'] = df['log_ratings_views_norm'].apply(views_mapping_3_std)

    # Add sentiment
    positive_ratings = {'Courageous', 'Beautiful', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring',
                        'Jaw-dropping', 'Persuasive'}
    negative_ratings = {'Confusing', 'Longwinded', 'OK', 'Obnoxious', 'Unconvincing'}

    def calculate_sentiment(ratings: list) -> float:
        """
        Returns a value from -1, 1 indicating negative/positive sentiment, by counting the positive and negative tags

        :param ratings: Ratings related to video ([{'name': 'Funny', 'count': 100}, {'name': 'Beautiful', 'count':10}, ...})
        :return: The sentiment
        """

        positive_count = np.sum([tag['count'] for tag in ratings if tag['name'] in positive_ratings])
        negative_count = np.sum([tag['count'] for tag in ratings if tag['name'] in negative_ratings])

        return (positive_count - negative_count) / (positive_count + negative_count)


    df['sentiment'] = df['ratings'].apply(ast.literal_eval).apply(calculate_sentiment)


    # Add normalized ratings
    def normalize_rating(ratings, name):
        rating_count = [t['count'] for t in ast.literal_eval(ratings) if t['name'] == name][0]
        return rating_count / np.sum([t['count'] for t in ast.literal_eval(ratings)])

    for rat in {*positive_ratings, *negative_ratings}:
        df[rat] = df['ratings'].apply(lambda ratings: normalize_rating(ratings, rat))
        df[f'log_{rat}'] = np.log(df[f'{rat}']+0.001)
        df[f'log_{rat}_norm'] = (df[f'log_{rat}'] - df[f'log_{rat}'].mean()) / df[f'log_{rat}'].std()
        df[f'log_{rat}_norm_cat'] = df[f'log_{rat}_norm'].apply(views_mapping_3_std)

        df[f"{rat}_views"] = df[rat]*np.log(df['views'])
        df[f"log_{rat}_views"] = np.log(df[f"{rat}_views"]+0.001)
        df[f'log_{rat}_views_norm'] = (df[f'log_{rat}_views'] - df[f'log_{rat}_views'].mean()) / df[f'log_{rat}_views'].std()
        df[f'log_{rat}_views_norm_cat'] = df[f'log_{rat}_views_norm'].apply(views_mapping_3_std)

    # Add total negative ratings
    negative_count = []
    for idx, row in df.iterrows():
        ratings = ast.literal_eval(row['ratings'])
        neg_sum = 0
        for rat in ratings:
            if rat['name'] in negative_ratings:
                neg_sum += rat['count']
        negative_count.append(neg_sum)

    df['negative_ratings'] = np.array(negative_count)
    df['log_negative_ratings'] = np.log(df['negative_ratings']+0.001)
    df['log_negative_ratings_norm'] = (df['log_negative_ratings'] - df['log_negative_ratings'].mean()) / df['log_negative_ratings'].std()
    df['log_negative_ratings_norm_cat'] = df['log_negative_ratings_norm'].apply(views_mapping_3_std)

    print(df)
    df.to_csv("../metadata/merged_metadata_popularity_std.csv", index=False)
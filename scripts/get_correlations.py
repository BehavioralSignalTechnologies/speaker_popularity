import ast

import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_rating_correlations(df, column, threshold=0.0):
    my_dict = {}
    for r in ratings:
        corr_mat = df[[column, r]].corr()
        if threshold > 0 and abs(corr_mat[r][0]) < threshold:
            my_dict[r] = ["-"]
        else:
            my_dict[r] = [round(corr_mat[r][0], 3)]
        # print(f"Views - {r}: {corr_mat[r][0]}")
    return my_dict


ratings = ['Funny', 'Beautiful', 'Ingenious', 'Courageous',
       'Longwinded', 'Confusing', 'Informative', 'Fascinating', 'Unconvincing',
       'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring']

merged_df = pd.read_csv("../metadata/merged_metadata_popularity_summaries.csv")

#### NORMALISE RATINGS
merged_df['ratings'] = merged_df['ratings'].apply(lambda x: ast.literal_eval(x))
speakers_mapped = {s: i for i, s in enumerate(merged_df["main_speaker"].unique())}
for row in merged_df.iterrows():
    speaker = row[1]["main_speaker"]
    rating_list = row[1]["ratings"]
    rating_sum = 0
    url = row[1]["url"]
    
    for r in rating_list:
        rating_sum += r["count"]
    # breakpoint()
    for r in rating_list:
        merged_df.at[row[0], r["name"]] = r["count"] / rating_sum
        merged_df.at[row[0], "speaker_id"] = speakers_mapped[speaker]


merged_df['transcript'] = merged_df['transcript'].fillna('')
merged_df['wc'] = merged_df['transcript'].apply(lambda x: len(x.split()))
merged_df['duration'] = merged_df['duration'] / 60
merged_df['wpm'] = merged_df['wc'] / merged_df['duration']

merged_df['comments_views'] = merged_df['comments'] / merged_df['views']
merged_df['views_comments'] = merged_df['views'] / merged_df['comments']

import numpy as np
age_dict ={
    "18 - 22": 0,
    "23 - 30": 1,
    "31 - 45": 2,
    "46 - 65": 3,
    np.nan: 'nan',
}
merged_df["age"] = merged_df["age"].apply(lambda x: age_dict[x])
print_dict = {}
corr_list = ["log_views_norm", "comments", "speaker_id", "languages", "wc", "wpm", "duration", "comments_views",
             "views_comments","age","speaker_turns","emotion_angry","emotion_happy","emotion_neutral","emotion_sad","positivity_neutral","positivity_negative","positivity_positive","strength_strong","strength_neutral","strength_weak"]

print_dict["/"] = corr_list
for c in corr_list:
    temp_dict = get_rating_correlations(merged_df, c, 0.2)
    for k, v in temp_dict.items():
        print_dict.setdefault(k, []).extend(v)
print(tabulate.tabulate(print_dict, headers='keys', tablefmt='rst'))
plt.figure(figsize=(13, 14))
sns.heatmap(merged_df[ratings + corr_list].corr(), annot=True)
plt.show()





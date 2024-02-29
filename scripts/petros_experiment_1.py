import ast
from collections import Counter
import tabulate
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def get_features(df):
    X_emb_mean = df['features_embedding_mean'].apply(ast.literal_eval).tolist()  # Features
    X_emb_std = df['features_embedding_std'].apply(ast.literal_eval).tolist()  # Features
    X_emb = np.concatenate([X_emb_mean, X_emb_std], axis=-1)


    # breakpoint()
    X_emb_transcript = np.array(df['embeddings_transcript_clean'].apply(ast.literal_eval).tolist())  # Features
    X_emb_tags = np.array(df['embeddings_tags'].apply(ast.literal_eval).tolist())  # Features

    post_cols = ['emotion_angry_mean', 'emotion_angry_90p', 'emotion_angry_std',
                 'emotion_happy_mean', 'emotion_happy_90p', 'emotion_happy_std',
                 'emotion_sad_mean', 'emotion_sad_90p', 'emotion_sad_std',
                 'emotion_neutral_mean', 'emotion_neutral_90p', 'emotion_neutral_std',
                 'strength_weak_mean', 'strength_weak_90p', 'strength_weak_std',
                 'strength_neutral_mean', 'strength_neutral_90p', 'strength_neutral_std',
                 'strength_strong_mean', 'strength_strong_90p', 'strength_strong_std',
                 'positivity_negative_mean', 'positivity_negative_90p',
                 'positivity_negative_std', 'positivity_neutral_mean',
                 'positivity_neutral_90p', 'positivity_neutral_std',
                 'positivity_positive_mean', 'positivity_positive_90p',
                 'positivity_positive_std', 'pauses_mean', 'pauses_std', 'pauses_10p',
                 'pauses_90p', 'turn_durations_mean', 'turn_durations_std',
                 'turn_durations_10p', 'turn_durations_90p']
    X_post = df[post_cols]  # Features

    X_mfccs_mean = df['mfccs_mean'].apply(ast.literal_eval).tolist()  # Features
    X_mfccs_std = df['mfccs_std'].apply(ast.literal_eval).tolist()  # Features
    X_mfccs = np.concatenate([X_mfccs_mean, X_mfccs_std], axis=-1)

    X_gen = np.array(df['gender_male_mean'].apply(lambda it: 1 if it > 0.5 else 0))
    X_gen = np.expand_dims(X_gen, axis=-1)

    # Normalize
    X_emb = (X_emb - X_emb.mean())/X_emb.std()
    X_emb_transcript = (X_emb_transcript - X_emb_transcript.mean())/X_emb_transcript.std()
    X_emb_tags = (X_emb_tags - X_emb_tags.mean())/X_emb_tags.std()
    X_post = (X_post - X_post.mean())/X_post.std()
    X_mfccs = (X_mfccs - X_mfccs.mean()) / X_mfccs.std()
    X_gen = (X_gen - X_gen.mean()) / X_gen.std()

    return X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags

def get_train_test_sets(df, target_col_cat):
    X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags = get_features(df)

    y_cat = df[target_col_cat]
    print(Counter(y_cat).most_common())

    (X_emb_train, X_emb_test, X_post_train, X_post_test, X_mfccs_train, X_mfccs_test, X_gen_train, X_gen_test, X_emb_transcript_train, X_emb_transcript_test, X_emb_tags_train, X_emb_tags_test,
     y_train_cat, y_test_cat) = train_test_split(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags, y_cat, test_size=0.2, random_state=42)
    return X_emb, X_post, X_mfccs, X_gen, y_cat, X_emb_train, X_emb_test, X_post_train, X_post_test, X_mfccs_train, X_mfccs_test, X_gen_train, X_gen_test, X_emb_transcript_train, X_emb_transcript_test, X_emb_tags_train, X_emb_tags_test, y_train_cat, y_test_cat


def cross_validation(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags, y_cat, visualize=False):
    X_post_gen = np.concatenate([X_post, X_gen], axis=-1)

    # Embeddings classifier
    emb_classifier = SVC(C=200)
    emb_scores = cross_val_score(emb_classifier, X_emb, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Embeddings Cross-validated F1: {emb_scores.mean()}")

    # Posterior classifier
    post_classifier = RandomForestClassifier()
    post_scores = cross_val_score(post_classifier, X_post, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Posteriors Cross-validated F1: {post_scores.mean()}")

    # Gender classifier
    gen_classifier = RandomForestClassifier()
    gen_scores = cross_val_score(gen_classifier, X_gen, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Gender Cross-validated F1: {gen_scores.mean()}")

    # Posterior w/ gender
    post_gen_classifier = RandomForestClassifier()
    post_gen_scores = cross_val_score(post_gen_classifier, X_post_gen, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Posteriors+gender Cross-validated F1: {post_gen_scores.mean()}")
    #
    # # MFCC classifier
    mfcc_classifier = SVC(C=200)
    mfcc_scores = cross_val_score(mfcc_classifier, X_mfccs, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"MFCC Cross-validated F1: {mfcc_scores.mean()}")

    # Text classifier
    text_classifier = SVC(C=200)
    text_scores = cross_val_score(text_classifier, X_emb_transcript, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Text Cross-validated F1: {text_scores.mean()}")

    # Text + tags classifier
    text_tags_classifier = SVC(C=200)
    text_tags_embeddings = np.concatenate([X_emb_transcript, X_emb_tags], axis=-1)
    text_tags_scores = cross_val_score(text_tags_classifier, text_tags_embeddings, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Text-Tags Cross-validated F1: {text_tags_scores.mean()}")

    # Audio + Text classifier
    audio_text_classifier = SVC(C=200)
    audio_text_embeddings = np.concatenate([X_emb, X_emb_transcript], axis=-1)
    audio_text_scores = cross_val_score(audio_text_classifier, audio_text_embeddings, y_cat, cv=5,
                                       scoring='f1_weighted', n_jobs=-1)
    print(f"Audio-Text Cross-validated F1: {audio_text_scores.mean()}")

    # # Audio + Tags classifier
    audio_tags_classifier = SVC(C=200)
    audio_tags_embeddings = np.concatenate([X_emb, X_emb_tags], axis=-1)
    audio_tags_scores = cross_val_score(audio_tags_classifier, audio_tags_embeddings, y_cat, cv=5,
                                       scoring='f1_weighted', n_jobs=-1)
    print(f"Audio-Tags Cross-validated F1: {audio_tags_scores.mean()}")

    # Audio + Text + tags classifier
    audio_text_tags_classifier = SVC(C=200)
    audio_text_tags_embeddings = np.concatenate([X_emb, X_emb_transcript, X_emb_tags], axis=-1)
    audio_text_tags_scores = cross_val_score(audio_text_tags_classifier, audio_text_tags_embeddings, y_cat, cv=5,
                                       scoring='f1_weighted', n_jobs=-1)
    print(f"Audio-Text-Tags Cross-validated F1: {audio_text_tags_scores.mean()}")

    # Dummy classifier
    dummy_clf = DummyClassifier(strategy='stratified')
    dummy_scores = cross_val_score(dummy_clf, X_emb, y_cat, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"Baseline Cross-validated F1: {dummy_scores.mean()}")

    # return (emb_scores.mean(), post_scores.mean(), gen_scores.mean(), post_gen_scores.mean(), mfcc_scores.mean(), text_scores.mean(), text_tags_scores.mean(),
    #         audio_text_scores.mean(), audio_tags_scores.mean(), audio_text_tags_scores.mean(), dummy_scores.mean())
    return (emb_scores.mean(), post_scores.mean(), gen_scores.mean(), post_gen_scores.mean(), mfcc_scores.mean(),
            text_scores.mean(), text_tags_scores.mean(), audio_text_scores.mean(), audio_tags_scores.mean(),
            audio_text_tags_scores.mean(), dummy_scores.mean())
    # return (text_scores.mean(), text_tags_scores.mean(), audio_text_scores.mean(), audio_text_tags_scores.mean(), dummy_scores.mean())


def train_test(df, target):
    (X_emb, X_post, X_mfccs, X_gen, y_cat, X_emb_train, X_emb_test, X_post_train, X_post_test, X_mfccs_train, X_mfccs_test, X_gen_train,
     X_gen_test, X_emb_transcript_train, X_emb_transcript_test, X_emb_tags_train, X_emb_tags_test, y_train_cat, y_test_cat) = get_train_test_sets(df, target)

    # Assuming 'y_test' contains the true class labels and 'y_pred' contains the predicted class labels
    emb_classifier = SVC(C=200)
    emb_classifier.fit(X_emb_train, y_train_cat)
    y_pred = emb_classifier.predict(X_emb_test)
    cm = confusion_matrix(y_test_cat, y_pred)

    # emb_classifier.fit(X_emb_train, y_train_cat)
    accuracy = accuracy_score(y_test_cat, y_pred)
    f1 = f1_score(y_test_cat, y_pred, average="weighted")
    print(f"Accuracy: {accuracy}, F1: {f1}")

    # Plotting the confusion matrix as a heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    # plt.show()
    plt.write_html('confusion.html')

def balance(df, column):
    # Display the initial group counts
    initial_counts = df[column].value_counts()

    # Find the minimum group size across all groups
    min_group_size = initial_counts.min()
    sampling_size = int(min_group_size + 0.2*min_group_size)

    # Create a new DataFrame to hold the balanced groups
    balanced_df = pd.DataFrame()

    # Loop over each group and randomly sample 'min_group_size' elements
    for group_name, group_data in df.groupby(column):
        try:
            balanced_group = group_data.sample(n=sampling_size, replace=False)
        except:
            balanced_group = group_data.sample(n=min_group_size, replace=False)
        balanced_df = pd.concat([balanced_df, balanced_group])
    return balanced_df

if __name__ == "__main__":
    positive_ratings = {'Courageous', 'Beautiful', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring',
                        'Jaw-dropping', 'Persuasive'}
    negative_ratings = {'Confusing', 'Longwinded', 'OK', 'Obnoxious', 'Unconvincing'}
    # positive_ratings_views = {f"{r}_views" for r in positive_ratings}
    # negative_ratings_views = {f"{r}_views" for r in negative_ratings}

    df_1 = pd.read_csv("../metadata/merged_metadata_popularity_features_std.csv")
    df_2 = pd.read_csv("../metadata/embeddings_transcript_clean.csv", usecols=['url', 'embeddings_transcript_clean'])
    df_3 = pd.read_csv("../metadata/embeddings_tags.csv", usecols=['url', 'embeddings_tags'])
    df_1['url'] = df_1['url'].astype(str).apply(str.strip)
    df_2['url'] = df_2['url'].astype(str).apply(str.strip)
    df_3['url'] = df_3['url'].astype(str).apply(str.strip)

    df_4 = pd.merge(df_1, df_2, on="url")

    df_4['url'] = df_4['url'].astype(str).apply(str.strip)

    df = pd.merge(df_4, df_3, on="url")
    df = df.dropna()


    cols = {'ratings_views', 'views', 'comments_per_view', *positive_ratings, *negative_ratings, 'negative_ratings'}
    scores = {'target': [], 'type': [], 'score': []}
    # X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags = get_features(df)
    print_dict = {}
    cols = sorted(list(cols))
    for col in cols:
        print("="*16)
        print(f"Predicting: {col}")
        balanced_df = balance(df, f"log_{col}_norm_cat")
        X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags = get_features(balanced_df)
        y_cat = balanced_df[f"log_{col}_norm_cat"]

        label_encoder = LabelEncoder()
        y_cat = label_encoder.fit_transform(y_cat)
        print(Counter(y_cat).most_common())
        # text_scores, text_tags_scores, audio_text_scores, audio_text_tags_scores, dummy_scores = cross_validation(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags, y_cat)

        (emb_cross_val_f1, post_cross_val_f1, gen_cross_val_f1, post_gen_cross_val_f1,
         mfcc_cross_val_f1, text_cross_val_f1, text_tags_cross_val_f1, audio_text_cross_val_f1,
         audio_tags_cross_val_f1, audio_text_tags_cross_val_f1,
         baseline_cross_val_f1) = cross_validation(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript,
                                                   X_emb_tags, y_cat)

        print_dict.setdefault("Task", []).append(col)
        print_dict.setdefault("Embeddings", []).append(emb_cross_val_f1)
        print_dict.setdefault("Posteriors", []).append(post_cross_val_f1)
        print_dict.setdefault("Gender", []).append(gen_cross_val_f1)
        print_dict.setdefault("PosterGender", []).append(post_gen_cross_val_f1)
        print_dict.setdefault("MFCC", []).append(mfcc_cross_val_f1)
        print_dict.setdefault("Text", []).append(text_cross_val_f1)
        print_dict.setdefault("Text+Tags", []).append(text_tags_cross_val_f1)
        print_dict.setdefault("Audio+Text", []).append(audio_text_cross_val_f1)
        print_dict.setdefault("Audio+Tags", []).append(audio_tags_cross_val_f1)
        print_dict.setdefault("Audio+Text+Tags", []).append(audio_text_tags_cross_val_f1)
        print_dict.setdefault("Dummy", []).append(baseline_cross_val_f1)


        # print_dict.setdefault("Text scores", []).append(text_scores)
        # print_dict.setdefault("Text+Tags scores", []).append(text_tags_scores)
        # print_dict.setdefault("Audio+Text scores", []).append(audio_text_scores)
        # print_dict.setdefault("Audio+Text+Tags scores", []).append(audio_text_tags_scores)
        # print_dict.setdefault("Dummy scores", []).append(dummy_scores)

        """
        (emb_cross_val_f1, post_cross_val_f1, gen_cross_val_f1, post_gen_cross_val_f1, mfcc_cross_val_f1, text_cross_val_f1, text_tags_cross_val_f1, audio_text_cross_val_f1,
         audio_tags_cross_val_f1, audio_text_tags_cross_val_f1, baseline_cross_val_f1) = cross_validation(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, X_emb_tags, y_cat)

        scores['target'].append(col)
        scores['type'].append('embeddings')
        scores['score'].append(emb_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('random baseline')
        scores['score'].append(baseline_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('posteriors')
        scores['score'].append(post_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('mfcc')
        scores['score'].append(mfcc_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('gender')
        scores['score'].append(gen_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('posteriors w/ gender')
        scores['score'].append(post_gen_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('embeddings_transcript')
        scores['score'].append(text_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('embeddings_text_tags')
        scores['score'].append(text_tags_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('audio_text')
        scores['score'].append(audio_text_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('audio_tags')
        scores['score'].append(audio_tags_cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('audio_text_tags')
        scores['score'].append(audio_text_tags_cross_val_f1)
        """
    print(tabulate.tabulate(print_dict, headers='keys', tablefmt='rst'))
    scores = pd.DataFrame(scores)
    scores.to_csv("scores.csv")
    sorted_scores = scores.sort_values(by="score")
    px.bar(sorted_scores, x="target", y="score", color="type", barmode='overlay').write_html("scores.html")


    train_test(df, 'log_Beautiful_norm_cat')


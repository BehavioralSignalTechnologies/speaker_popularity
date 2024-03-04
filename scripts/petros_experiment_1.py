import ast
import concurrent
import os
from collections import Counter
import tabulate
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_features(df):
    X_emb_mean = df['features_embedding_mean'].apply(ast.literal_eval).tolist()  # Features
    X_emb_std = df['features_embedding_std'].apply(ast.literal_eval).tolist()  # Features
    X_emb = np.concatenate([X_emb_mean, X_emb_std], axis=-1)

    X_emb_transcript = np.array(df['embeddings'].apply(ast.literal_eval).tolist())  # Features

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
    X_post = df[post_cols].to_numpy()  # Features
    print(X_post.shape)

    X_mfccs_mean = df['mfccs_mean'].apply(ast.literal_eval).tolist()  # Features
    X_mfccs_std = df['mfccs_std'].apply(ast.literal_eval).tolist()  # Features
    X_mfccs = np.concatenate([X_mfccs_mean, X_mfccs_std], axis=-1)

    X_gen = np.array(df['gender_male_mean'].apply(lambda it: 1 if it > 0.5 else 0))
    X_gen = np.expand_dims(X_gen, axis=-1)

    return X_emb, X_post, X_mfccs, X_gen, X_emb_transcript

def get_train_test_sets(df, target_col_cat):
    X_emb, X_post, X_mfccs, X_gen, X_emb_transcript = get_features(df)

    y_cat = df[target_col_cat]
    print(Counter(y_cat).most_common())

    (X_emb_train, X_emb_test, X_post_train, X_post_test, X_mfccs_train, X_mfccs_test, X_gen_train, X_gen_test, X_emb_transcript_train, X_emb_transcript_test,
     y_train_cat, y_test_cat) = train_test_split(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, y_cat, test_size=0.2, random_state=42)
    return X_emb, X_post, X_mfccs, X_gen, y_cat, X_emb_train, X_emb_test, X_post_train, X_post_test, X_mfccs_train, X_mfccs_test, X_gen_train, X_gen_test, X_emb_transcript_train, X_emb_transcript_test, y_train_cat, y_test_cat


def extreme_errors(y_true, y_pred):
    y_true_ids = set(np.where(y_true != "medium")[0])
    y_pred_ids = set(np.where(y_pred != "medium")[0])
    ids = np.array(list(y_true_ids.intersection(y_pred_ids)))

    if len(ids) == 0:
        return 0.0

    y_pred_hl = y_pred[ids]
    y_true_hl = y_true[ids]

    return accuracy_score(y_true_hl, y_pred_hl)


def calculate_scores(col, name, clf, X, y, classes):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    precision = {cls: [] for cls in classes}
    recall = {cls: [] for cls in classes}
    auc = {cls: [] for cls in classes}
    f1_macro = []
    f1_weighted = []
    ext_err = []

    for idx, (train_index, test_index) in enumerate(cv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardize fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)

        if idx == 0:
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=["low", "medium", "high"])
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=True, yticklabels=True)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            plt.savefig(f"../results/cm_{col}_{name}.png")
            plt.close()

        f1_macro.append(f1_score(y_test, y_pred, average="macro"))
        f1_weighted.append(f1_score(y_test, y_pred, average="weighted"))
        ext_err.append(extreme_errors(y_test, y_pred))

        aucs = roc_auc_score(y_test, y_prob, multi_class='ovr', average=None, labels=classes)

        for idx, cls in enumerate(classes):
            auc[cls].append(aucs[idx])
            precision[cls].append(precision_score(y_test, y_pred, labels=[cls], average=None))
            recall[cls].append(recall_score(y_test, y_pred, labels=[cls], average=None))


    # Average the scores across folds for each class
    averaged_scores = {
        **{f"precision_{cls}": np.mean(precision[cls]) for cls in classes},
        **{f"recall_{cls}": np.mean(recall[cls]) for cls in classes},
        **{f"auc_{cls}": np.mean(auc[cls]) for cls in auc},
        'f1_macro': np.mean(f1_macro),
        'f1_weighted': np.mean(f1_weighted),
        'extreme_errors': np.mean(ext_err)
    }
    return averaged_scores


def evaluate_classifier(col, name, X, y, classes):
    if name in ['Embeddings', 'MFCC', 'Text', 'Audio-Text']:
        clf = SVC(C=200, probability=True)
    elif name == 'Baseline':
        clf = DummyClassifier(strategy='stratified')
    else:
        clf = RandomForestClassifier()

    scores = calculate_scores(col, name, clf, X, y, classes)
    return scores


def cross_validation_with_kfold_split_and_class_metrics(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, cols):

    # Helper function to calculate scores
    X_post_gen = np.concatenate([X_post, X_gen], axis=-1)
    audio_text_embeddings = np.concatenate([X_emb, X_emb_transcript], axis=-1)

    X_data = {
        'Embeddings': X_emb,
        'Posteriors': X_post,
        'Gender': X_gen,
        'Posteriors+gender': X_post_gen,
        'MFCC': X_mfccs,
        'Text': X_emb_transcript,
        'Audio-Text': audio_text_embeddings,
        'Baseline': X_emb
    }

    results = {col: {} for col in cols}
    print(f"Running on {os.cpu_count()} cores")

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}
        for col in cols:
            print("=" * 16)
            print(f"Predicting: {col}")

            y_cat = df[f"log_{col}_norm_cat"].to_numpy()
            print(Counter(y_cat).most_common())

            # Assuming y_cat contains the class labels as integers
            classes = np.unique(y_cat)

            for name in X_data:
                futures[executor.submit(evaluate_classifier, col, name, X_data[name], y_cat, classes)] = f"{col}__{name}"

        # Initialize the tqdm progress bar
        pbar = tqdm(total=len(futures), desc="Classifying")

        for future in concurrent.futures.as_completed(futures):
            col, name = futures[future].split("__")
            try:
                results[col][name] = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (name, exc))
            finally:
                # Manually update the progress bar
                pbar.update(1)

    return results


def train_test(df, target):
    (X_emb, X_post, X_mfccs, X_gen, y_cat, X_emb_train, X_emb_test, X_post_train, X_post_test, X_mfccs_train, X_mfccs_test, X_gen_train,
     X_gen_test, X_emb_transcript_train, X_emb_transcript_test, y_train_cat, y_test_cat) = get_train_test_sets(df, target)

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
    plt.show()


if __name__ == "__main__":
    positive_ratings = {'Courageous', 'Beautiful', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring',
                        'Jaw-dropping', 'Persuasive'}
    negative_ratings = {'Confusing', 'Longwinded', 'OK', 'Obnoxious', 'Unconvincing'}
    # positive_ratings_views = {f"{r}_views" for r in positive_ratings}
    # negative_ratings_views = {f"{r}_views" for r in negative_ratings}

    df_1 = pd.read_csv("../metadata/merged_metadata_popularity_features_std.csv")
    df_2 = pd.read_csv("../metadata/embeddings_transcript_clean.csv")
    df_1['url'] = df_1['url'].astype(str).apply(str.strip)
    df_2['url'] = df_2['url'].astype(str).apply(str.strip)

    df = pd.merge(df_1, df_2, on="url")
    df = df.dropna()


    cols = {'ratings_views', 'views', 'comments_per_view', *positive_ratings, *negative_ratings, 'negative_ratings'}
    scores = {'target': [], 'type': [], 'metric': [], 'value': []}
    X_emb, X_post, X_mfccs, X_gen, X_emb_transcript = get_features(df)
    print_dict = {}
    results_cross_val = cross_validation_with_kfold_split_and_class_metrics(X_emb, X_post, X_mfccs, X_gen, X_emb_transcript, cols)

    for col in results_cross_val.keys():
        for name, results in results_cross_val[col].items():
            for metric, value in results.items():
                scores['target'].append(col)
                scores['type'].append(name)
                scores['metric'].append(metric)
                scores['value'].append(value)


    scores = pd.DataFrame(scores)
    scores.to_csv("../results/scores.csv")
    sorted_scores = scores.sort_values(by="value")
    px.bar(sorted_scores, x="target", y="value", color="type", facet_row="metric", barmode='overlay').show()

    # train_test(df, 'log_Beautiful_norm_cat')


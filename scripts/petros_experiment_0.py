from collections import Counter

import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

features_metadata_df = pd.read_csv("../metadata/merged_metadata_popularity_features.csv")


def train_test(target_col_cat='log_views_norm_cat', visualize=False):
    cols = ['emotion_angry_mean', 'emotion_angry_90p', 'emotion_angry_std',
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

    X = features_metadata_df.dropna()[cols]  # Features
    y_cat = features_metadata_df.dropna()[target_col_cat]
    print(Counter(y_cat).most_common())

    X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train_cat)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test_cat, y_pred)
    f1 = f1_score(y_test_cat, y_pred, average="weighted")
    print(f"Accuracy: {accuracy}, F1: {f1}")

    scores = cross_val_score(classifier, X, y_cat, cv=5, scoring='f1_weighted')
    print(f"Cross-validated F1: {scores.mean()}")

    dummy_clf = DummyClassifier(strategy='stratified')
    dummy_clf.fit(X_train, y_train_cat)
    dummy_scores = cross_val_score(dummy_clf, X, y_cat, cv=5, scoring='f1_weighted')
    print(f"Baseline Cross-validated F1: {dummy_scores.mean()}")


    if visualize:
        # Assuming 'y_test' contains the true class labels and 'y_pred' contains the predicted class labels
        cm = confusion_matrix(y_test_cat, y_pred)

        # Plotting the confusion matrix as a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=True, yticklabels=True)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    return scores.mean(), dummy_scores.mean()

if __name__ == "__main__":
    positive_ratings = {'Courageous', 'Beautiful', 'Fascinating', 'Funny', 'Informative', 'Ingenious', 'Inspiring',
                        'Jaw-dropping', 'Persuasive'}
    negative_ratings = {'Confusing', 'Longwinded', 'OK', 'Obnoxious', 'Unconvincing'}
    cols = {'views', 'comments_per_view', *positive_ratings, *negative_ratings, 'negative_ratings'}
    scores = {'target': [], 'type': [], 'score': []}
    for col in cols:
        print("=" * 16)
        print(f"Predicting: {col}")
        cross_val_f1, baseline_cross_val_f1 = train_test(f"log_{col}_norm_cat")

        scores['target'].append(col)
        scores['type'].append('posteriors')
        scores['score'].append(cross_val_f1)

        scores['target'].append(col)
        scores['type'].append('random baseline')
        scores['score'].append(baseline_cross_val_f1)

    df = pd.DataFrame(scores)
    sorted_scores = df.sort_values(by="score")
    px.bar(sorted_scores, x="target", y="score", color="type", barmode='overlay').show()

    train_test(f"log_Informative_norm_cat", visualize=True)
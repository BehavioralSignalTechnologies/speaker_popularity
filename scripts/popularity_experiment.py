import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score,
                             precision_score, recall_score)

age_dict ={
    "18 - 22": 0,
    "23 - 30": 1,
    "31 - 45": 2,
    "46 - 65": 3,
    np.nan: -1,
    "0": -1,
    0: -1,
}

gender_dict ={
    'male': 0,
    'female': 1,
    0: -1
}

def compute_class_performance(confusion_matrix, n_class):
    """!
    @brief Compute recall, precision and F1 for the n-th class,
           given the confusion matrix.

    @param n_class (\a int) The n-th class.
    @param confusion_matrix (\a numpy array) The confusion matrix.

    @returns \b performance (\a dict) Dictionary in the following
             format: {"recall": recall, "precision": precision,
             "f1": f1}
    """
    eps = np.finfo(float).eps
    correct = float(confusion_matrix[n_class, n_class])

    recall = correct / (np.sum(confusion_matrix[n_class, :]) + eps)
    precision = correct / (np.sum(confusion_matrix[:, n_class]) + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)

    return {"recall": recall, "precision": precision, "f1": f1,
            "correct": correct}

def normalize_matrix(matrix, mean=None, std=None):
    """Use mean and standard deviation to normalize a feature matrix

    Args:
        matrix (numpy.array): 2D matrix whose rows are data-points and whose
            columns are features
        mean (numpy.array, optional): 1D matrix with column means
        std (numpy.array, optional): 1D matrix with column standard deviations

    Returns:
        tuple: entry 0 is the normalized matrix, entry 1 is the column means,
        entry 2 is the column standard deviations
    """
    if std is None:
        std = matrix.std(axis=0)
    if mean is None:
        mean = matrix.mean(axis=0)

    norm_matrix = ((matrix - mean) / (std + 1e-10))
    return norm_matrix, mean, std

def compute_scores_from_cm(cm):
    """!
    @brief Compute average scores from confusion matrix
    """
    f1_scores = []
    recall = []
    precision = []
    correct = []

    for i in range(cm.shape[0]):
        if np.all(cm[i, :] == 0) and np.all(cm[:, i] == 0):
            continue
        perf = compute_class_performance(cm, i)
        f1_scores.append(perf["f1"])
        recall.append(perf["recall"])
        precision.append(perf["precision"])
        correct.append(perf["correct"])

    pr = np.mean(precision)
    f1 = np.mean(f1_scores)
    uar = np.mean(recall)
    acc = np.sum(correct) / np.sum(cm)

    return {"f1": f1, "acc": acc, "uar": uar, "precision": pr}

def normalize_confusion_matrix(cm):
    """!
    @brief Normalize confusion matrix per row samples.
    """
    total = cm.sum(axis=1) + np.finfo(float).eps
    cm = cm / total[:, np.newaxis]
    return cm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input file",
                        default="../metadata/merged_metadata_popularity_summaries.csv")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    df = pd.read_csv(args.input)
    labels_df = df["log_views_norm_cat"].to_numpy()

    #### THIS NEEDS SOME EXPLORATION TO AVOID FILLING ALL NAN WITH ZEROS WHICH MAY BE WRONG #####
    df.fillna(0, inplace=True)

    # df["age"] = df["age"].apply(lambda x: age_dict[x])
    # df["gender"] = df["gender"].apply(lambda x: gender_dict[x])
    df["norm_turns"] = df["speaker_turns"] / df["duration"]
    features = df.loc[:, 'emotion_angry':'norm_turns'].to_numpy()
    model = xgb.XGBClassifier()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_df)

    kf = KFold(n_splits=10)
    cm, f1, acc, precision, uar= [], [], [], [], []
    for i, (train_index, test_index) in enumerate(kf.split(features)):
        train_set = features[train_index]
        train_labels = labels[train_index]
        train_set_norm, mean, std = normalize_matrix(train_set)

        model.fit(train_set_norm, train_labels)

        test_labels = labels[test_index]
        test_set = features[test_index]
        test_set_norm, _, _ = normalize_matrix(test_set, mean, std)
        pred_labels = model.predict(test_set_norm)

        cm.append(confusion_matrix(test_labels, pred_labels))
        acc.append(accuracy_score(test_labels, pred_labels))
        f1.append(f1_score(test_labels, pred_labels, average='macro'))
        uar.append(recall_score(test_labels, pred_labels, average='macro'))
        precision.append(precision_score(test_labels, pred_labels, average='macro'))
    print("\nClasses: {}".format(' '.join([str(m) for m
                                           in model.classes_])))
    print("Confusion matrix:\n{}".format(sum(cm)))
    print("Acc: {}".format(round(np.mean(acc), 3)))
    print("Precision: {}".format(round(np.mean(precision), 3)))
    print("Recall: {}".format(round(np.mean(uar), 3)))
    print("F1: {}".format(round(np.mean(f1), 3)))
    results = compute_scores_from_cm(normalize_confusion_matrix(sum(cm)))
    print("Norm F1: {}".format(results["f1"]))
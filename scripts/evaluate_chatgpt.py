import ast
import json
import tabulate
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score,
                             precision_score, recall_score, top_k_accuracy_score)

ratings = ['Funny', 'Beautiful', 'Ingenious', 'Courageous',
           'Longwinded', 'Confusing', 'Informative', 'Fascinating', 'Unconvincing',
           'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring']

top_k = 5

if __name__ == '__main__':
    chatgpt_file = "chatgpt_responses2_bak.json"
    with open(chatgpt_file, "r") as f:
        chatgpt_responses = json.load(f)
    df = pd.read_csv("../metadata/merged_metadata_popularity_features.csv")
    gt_list = {}
    pred_list = {}
    df['ratings'] = df['ratings'].apply(ast.literal_eval)
    for index, row in df.iterrows():
        filename = row["filename"]
        ratings_sorted = sorted(row['ratings'], key=lambda x: x['count'], reverse=True)
        rating_list = [r['name'] for r in ratings_sorted][:top_k]
        if filename not in chatgpt_responses:
            continue
        chatgpt_labels = chatgpt_responses[filename]["Labels"]
        for r in ratings:
            if r in rating_list:
                gt_list.setdefault(r, []).append(1)
            else:
                gt_list.setdefault(r, []).append(0)
            if r in chatgpt_labels:
                pred_list.setdefault(r, []).append(1)
            else:
                pred_list.setdefault(r, []).append(0)
    print_dict = {}
    for rate in ratings:
        # acc_score = accuracy_score(gt_list[rate], pred_list[rate])
        # if acc_score == 0.0:
        #     continue

        print(f"Rating: {rate}")
        print(f"Accuracy: {accuracy_score(gt_list[rate], pred_list[rate])}")
        print(f"Precision: {precision_score(gt_list[rate], pred_list[rate])}")
        print(f"Recall: {recall_score(gt_list[rate], pred_list[rate])}")
        print(f"F1: {f1_score(gt_list[rate], pred_list[rate])}")
        # print(f"Top-{top_k} accuracy: {top_k_accuracy_score(gt_list[rate], pred_list[rate])}")
        print(f"\nConfusion matrix: {confusion_matrix(gt_list[rate], pred_list[rate])}")
        print("\n")
        print_dict.setdefault("Rate", []).append(rate)
        print_dict.setdefault("Accuracy", []).append(accuracy_score(gt_list[rate], pred_list[rate]))
        print_dict.setdefault("Precision", []).append(precision_score(gt_list[rate], pred_list[rate]))
        print_dict.setdefault("Recall", []).append(recall_score(gt_list[rate], pred_list[rate]))
        print_dict.setdefault("F1", []).append(f1_score(gt_list[rate], pred_list[rate]))
    print(tabulate.tabulate(print_dict, headers='keys', tablefmt='rst'))



    # for key, value in chatgpt_responses.items():

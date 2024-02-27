import ast
import json
import os 
import librosa
import numpy as np
import argparse
import plotly.express as px
import pandas as pd
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Summarize a video')
    parser.add_argument('--in_json', type=str, help='path to json file with modeling-api results', required=True)
    parser.add_argument('--output', type=str, help='path to output video file', required=True)
    args = parser.parse_args()
    return args

tasks = {
    'emotion': ['angry', 'happy', 'sad', 'neutral'],
    'strength': ['weak', 'neutral', 'strong'],
    'positivity': ['negative', 'neutral', 'positive'],
    'gender': ['male', 'female']
    # TODO: add more classification tasks to be used here
}

def run_embedding_aggregation(input_json):
    key = list(input_json.keys())[0]

    selected_res = {}

    for pred in input_json[key]:
        if pred['id'] not in selected_res:
            selected_res[pred['id']] = {'st': pred['startTime'], 'et': pred['endTime']}
        if pred["task"] == 'features':
            if pred['id'] in selected_res:
                selected_res[pred['id']]['features'] = np.array(ast.literal_eval(pred['embedding']))

    # get average strength and emotion:
    # mean posteriors:
    mean_embeddings = {'vals': [], 'durations': []}
    for utterance in selected_res:
        if 'features' in selected_res[utterance]:
            mean_embeddings['vals'].append(selected_res[utterance]['features'])
            mean_embeddings['durations'].append(float(selected_res[utterance]['et']) - float(selected_res[utterance]['st']))

    # calculate mean embedding:
    def weighted_avg_and_std(values: np.ndarray, weights):
        """
        Return the weighted average and standard deviation.

        They weights are in effect first normalized so that they
        sum to 1 (and so they must not all be 0).

        values, weights -- NumPy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights, axis=0)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights, axis=0)
        return (average, np.sqrt(variance))

    mean_embeddings['mean'], mean_embeddings['std'] = weighted_avg_and_std(np.array(mean_embeddings['vals']), mean_embeddings['durations'])

    # delete vals:
    del mean_embeddings['vals']
    del mean_embeddings['durations']

    return mean_embeddings

def run_aggregation(input_json):
    key = list(input_json.keys())[0]

    selected_res = {}

    for pred in input_json[key]:
        if pred['id'] not in selected_res:
            selected_res[pred['id']] = {'st': pred['startTime'], 'et': pred['endTime']}
        for task in tasks:
            if pred["task"] == task:
                if pred['id'] in selected_res:
                    selected_res[pred['id']][task] = {item['label']: item['posterior'] for item in pred['prediction']}

    # get average strength and emotion:
    # mean posteriors:
    mean_posteriors = {}
    pauses = []
    durations = []
    count_utts = 0
    for utterance in selected_res:
        count_utts += 1
        if count_utts > 1:
            pauses.append(float(selected_res[utterance]['st']) - float(selected_res[prev_utterance]['et']))
        durations.append(float(selected_res[utterance]['et']) - float(selected_res[utterance]['st']))
        prev_utterance = utterance

        for task in tasks:
            if task not in mean_posteriors:
                mean_posteriors[task] = {}
            if task in selected_res[utterance]:
                for label in tasks[task]:
                    if label in selected_res[utterance][task]:
                        if label not in mean_posteriors[task]:
                            mean_posteriors[task][label] = {'vals': []}
                        mean_posteriors[task][label]['vals'].append(float(selected_res[utterance][task][label]))

    # calculate statistics on behavioral posteriors:
    for task in mean_posteriors:
        for label in mean_posteriors[task]:
            mean_posteriors[task][label]['mean'] = np.mean(mean_posteriors[task][label]['vals'])
            mean_posteriors[task][label]['90p'] = np.percentile(mean_posteriors[task][label]['vals'], 90)
            mean_posteriors[task][label]['std'] = np.std(mean_posteriors[task][label]['vals'])
            # delete vals:
            del mean_posteriors[task][label]['vals']

    mean_posteriors['pauses'] = {'mean': np.mean(pauses), 'std': np.std(pauses), '10p': np.percentile(pauses, 10),
                                 '90p': np.percentile(pauses, 90)}
    mean_posteriors['turn_durations'] = {'mean': np.mean(durations), 'std': np.std(durations), '10p': np.percentile(durations, 10),
                                         '90p': np.percentile(durations, 90)}
    return mean_posteriors


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.in_json) as f:
        res = json.load(f)

    mean_posteriors = run_aggregation(res)

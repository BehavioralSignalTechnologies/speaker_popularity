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
    # TODO: add more classification tasks to be used here
}

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
            # delete vals:
            del mean_posteriors[task][label]['vals']

    mean_posteriors['pauses'] = {'mean': np.mean(pauses), '10p': np.percentile(pauses, 10),
                                 '90p': np.percentile(pauses, 90)}
    mean_posteriors['turn_durations'] = {'mean': np.mean(durations), '10p': np.percentile(durations, 10),
                                         '90p': np.percentile(durations, 90)}
    return mean_posteriors


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.in_json) as f:
        res = json.load(f)

    mean_posteriors = run_aggregation(res)

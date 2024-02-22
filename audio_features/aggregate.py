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

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.in_json) as f:
        res = json.load(f)

    print(len(res))

    key = list(res.keys())[0]

    selected_res = {}

    for pred in res[key]:
        print(pred)
        if pred['id'] not in selected_res:
            selected_res[pred['id']] = {'st': pred['startTime'], 'et': pred['endTime']}
        for task in tasks:
            if pred["task"] == task:
                if pred['id'] in selected_res:
                    selected_res[pred['id']][task] = {item['label']: item['posterior'] for item in pred['prediction']}

    # get average strength and emotion:
    # mean posteriors:
    mean_posteriors = {}

    for utterance in selected_res:
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

    print(mean_posteriors)

    # compute pauses duration:
    pauses = []
    for i in range(len(selected_res) - 1):
        print(i)
        pauses.append(selected_res[i+1]['st'] - selected_res[i]['et'])
    print(pauses)
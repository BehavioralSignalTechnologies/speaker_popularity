import os
import ast
import re
import tqdm
import json
import openai
import pandas as pd

OPENAI_API_KEY = "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" #nassos

ratings = ['Funny', 'Beautiful', 'Ingenious', 'Courageous',
       'Longwinded', 'Confusing', 'Informative', 'Fascinating', 'Unconvincing',
       'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring', 'Controversial']

def generate_summary(examples, description, text):
    openai.api_key = OPENAI_API_KEY

    response = openai.chat.completions.create(
        response_format={ "type": "json_object" },
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are an expert annotator and  i want you to annotate the following on the given text as an output JSON."
                                          f"The text belongs to a TED talk with a single speaker on a given topic. Annotations in text like '(Laughter)' means that the audience was laughing. \n"
              f"Annotation labels:\n"
              f"1. Talk topic \n"
              f"2. I want you to choose from the following label list the labels which describe the talk. The list should be sorted by relevance. Choose a maximum of 3 labels. Please take care to"
              f"not select conflicting labels simultaneously like 'Persuasive' and 'Unconvincing'. Labels : {', '.join(ratings)} \n"
              f"3. Sentiment: [Negative, Neutral, Positive] depending on the sentiment of the words used in the topic\n"
              f"4. Description versus content relevant -1 for deceptive, 0 for not relevant, 1 for relevant\n"
              f"5. Describe the speaker by using at most 3 personality trait labels, for example 'arrogant' or 'disrespectful', 'fair'\n"},
            {"role": "user", "content": examples[0][0]},
            {"role": "assistant", "content": examples[0][1]},
            {"role": "user", "content": examples[1][0]},
            {"role": "assistant", "content": examples[1][1]},
            {"role": "user", "content": examples[2][0]},
            {"role": "assistant", "content": examples[2][1]},
            {"role": "user", "content": "Annotation sample:"
              f"Description: \n{description}"
              f"Text:\n"
              f"\n{text}"},
        ],
        temperature=0.3,
        n=1
        # stop="." ,
        # frequency_penalty=0.0,
        # presence_penalty=0.0
    )

    response_json = response.choices[0].message.content
    return ast.literal_eval(response_json)


if __name__ == '__main__':
    filename = "../metadata/merged_metadata_popularity.csv"
    df = pd.read_csv(filename)
    responses = {}

    output_file = "chatgpt_responses2.json"
    existing_responses = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            existing_responses = json.load(file)

    examples = []
    try:
        for index, row in tqdm.tqdm(df.iterrows()):
            text = row["transcript"]
            description = row["description"]
            filename = row["filename"]
            if filename in existing_responses:
                print(f"Response for file already exists: {filename}")
                continue
            ratings_row = sorted(ast.literal_eval(row["ratings"]), key=lambda x: x.get("count"), reverse=True)
            rating_labels = [r["name"] for r in ratings_row][:5]
            if index < 3:
                if index == 0:
                    talk_topic = "Sugar raises obesity"
                if index == 1:
                    talk_topic = "Augmented reality mapping technology"
                if index == 2:
                    talk_topic = "Smarter policies using Behavioral Economics"
                example_1 = (f"\nExample {index + 1}: "
                             f"Description: {description}'\n"
                             f"Text:\n {text}\n",
                            "{"
                             f"Talk topic: {talk_topic},\n"
                             f"Labels: {rating_labels},\n"
                             "Sentiment: Positive,\n"
                             "Description versus content relevance: 1,\n"
                             "Speaker personality traits: ['Humorous', 'Engaging', 'Knowledgeable'],\n"
                             "}")
                examples.append(example_1)
            else:
                responses[filename] = generate_summary(examples, description, text)
    except (Exception, KeyboardInterrupt) as e:
        print("Exception occured", e)
    with open(output_file, 'w+') as file:
        json.dump(responses, file)

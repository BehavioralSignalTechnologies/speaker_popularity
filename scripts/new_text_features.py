import pandas as pd
import tqdm
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
import textstat
import numpy as np
from openai import OpenAI
import json

# -------------------------------
# CONFIG
# -------------------------------
CSV_INPUT = "../metadata/merged_metadata.csv"
CSV_OUTPUT = "../metadata/enriched_features_transcript.csv"
TRANSCRIPT_COLUMN = "transcript"
OPENAI_API_KEY = ""  # Add your API key here
LLM_MODEL_NAME = "gpt-4"  # or gpt-3.5-turbo

# -------------------------------
# Initialize OpenAI client
# -------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# Helper functions
# -------------------------------
def clean_transcript(transcript):
    return re.sub(r'\(.*?\)', '', transcript)

def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if len(words) > 0 else 0

def clarity_score(text):
    return textstat.flesch_reading_ease(text)

def get_sentiment_embedding_and_posterior(text, tokenizer, embedding_model, classifier_model, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        # Embedding
        embedding_outputs = embedding_model(**inputs)
        embedding = embedding_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        # Sentiment logits
        classifier_outputs = classifier_model(**inputs)
        logits = classifier_outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().numpy()
        pred_label_idx = probs.argmax()
        pred_label = classifier_model.config.id2label[pred_label_idx]
        pred_prob = probs[pred_label_idx]
    return embedding, pred_label, pred_prob, probs

def prompt_task_features(transcript):
    if not transcript.strip():
        return None
    prompt = f"""
    Extract features from this transcript that are predictive of the video’s popularity.
    These features should help predict:
    1. View count (V)
    2. Comment ratio (C/V)
    3. Ratings per view
    4. Negative tag ratio (p4)
    5. Counts of viewer-assigned positive and negative tags

    For each feature below, rate it from 0 (low) to 10 (high) and output strictly as JSON:

    - engagement
    - clarity
    - emotional_intensity
    - storytelling
    - positive_tag_signal (likelihood of positive tags: Beautiful, Courageous, Funny, Fascinating, Informative, Inspiring, Ingenious, Jaw-dropping, OK, Persuasive)
    - negative_tag_signal (likelihood of negative tags: Longwinded, Confusing, Unconvincing, Obnoxious)

    Transcript:
    \"\"\"{transcript}\"\"\"
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        return json.loads(content.strip())
    except Exception as e:
        print("Error for transcript:", transcript[:100])
        print(e)
        return None

# -------------------------------
# Load CSV and clean transcripts
# -------------------------------
df = pd.read_csv(CSV_INPUT)
df[TRANSCRIPT_COLUMN] = df[TRANSCRIPT_COLUMN].apply(clean_transcript)

# -------------------------------
# Initialize sentiment models
# -------------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

embedding_model.eval()
classifier_model.eval()
class_names = [classifier_model.config.id2label[i] for i in range(classifier_model.config.num_labels)]

# -------------------------------
# Process transcripts
# -------------------------------
sentiment_embeddings = []
sentiment_labels = []
sentiment_scores = []
posterior_dicts = []
lex_divs = []
clarities = []

task_engagements = []
task_clarities = []
task_emotional_intensities = []
task_storytelling_qualities = []
task_positive_tags = []
task_negative_tags = []

for tr in tqdm.tqdm(df[TRANSCRIPT_COLUMN], total=len(df)):
    if tr == "":
        sentiment_embeddings.append(pd.NA)
        sentiment_labels.append(pd.NA)
        sentiment_scores.append(pd.NA)
        posterior_dicts.append({cls: pd.NA for cls in class_names})
        lex_divs.append(pd.NA)
        clarities.append(pd.NA)
        task_engagements.append(pd.NA)
        task_clarities.append(pd.NA)
        task_emotional_intensities.append(pd.NA)
        task_storytelling_qualities.append(pd.NA)
        task_positive_tags.append(pd.NA)
        task_negative_tags.append(pd.NA)
        continue

    try:
        # Sentiment features
        emb, label, score, posteriors = get_sentiment_embedding_and_posterior(
            tr, tokenizer, embedding_model, classifier_model
        )
        sentiment_embeddings.append(emb)
        sentiment_labels.append(label)
        sentiment_scores.append(score)
        posterior_dicts.append({cls: posteriors[i] for i, cls in enumerate(class_names)})

        # Lexical features
        lex_divs.append(lexical_diversity(tr))
        clarities.append(clarity_score(tr))

        # Task-aware LLM features
        task_feats = prompt_task_features(tr)
        if task_feats:
            task_engagements.append(task_feats.get("engagement", None))
            task_clarities.append(task_feats.get("clarity", None))
            task_emotional_intensities.append(task_feats.get("emotional_intensity", None))
            task_storytelling_qualities.append(task_feats.get("storytelling", None))
            task_positive_tags.append(task_feats.get("positive_tag_signal", None))
            task_negative_tags.append(task_feats.get("negative_tag_signal", None))
        else:
            task_engagements.append(None)
            task_clarities.append(None)
            task_emotional_intensities.append(None)
            task_storytelling_qualities.append(None)
            task_positive_tags.append(None)
            task_negative_tags.append(None)

    except Exception as e:
        print(tr[:100])
        raise e

# -------------------------------
# Save results
# -------------------------------
df['sentiment_embedding'] = [",".join(map(str, emb)) if isinstance(emb, np.ndarray) else pd.NA for emb in sentiment_embeddings]
df['sentiment_label'] = sentiment_labels
df['sentiment_score'] = sentiment_scores
df['lexical_diversity'] = lex_divs
df['clarity'] = clarities

# Posterior columns
for cls in class_names:
    df[f'posterior_{cls}'] = [d[cls] for d in posterior_dicts]

# Task-aware features
df['task_engagement'] = task_engagements
df['task_clarity'] = task_clarities
df['task_emotional_intensity'] = task_emotional_intensities
df['task_storytelling'] = task_storytelling_qualities
df['task_positive_tag_signal'] = task_positive_tags
df['task_negative_tag_signal'] = task_negative_tags

# Save final CSV
columns_to_save = ['url', 'sentiment_embedding', 'sentiment_label', 'sentiment_score'] + \
                  [f'posterior_{cls}' for cls in class_names] + \
                  ['lexical_diversity', 'clarity', 'task_engagement', 'task_clarity',
                   'task_emotional_intensity', 'task_storytelling', 'task_positive_tag_signal',
                   'task_negative_tag_signal']

df[columns_to_save].to_csv(CSV_OUTPUT, index=False)

print("Done! All enriched features saved.")

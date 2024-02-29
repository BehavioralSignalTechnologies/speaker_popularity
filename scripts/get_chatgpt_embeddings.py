import pandas as pd
import openai
import tqdm



# OPENAI_API_KEY = "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" # antonia
OPENAI_API_KEY = "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" #nassos

def generate_embeddings(text):
    # openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input=[text], model='text-embedding-3-large').data[0].embedding
    return embedding

if __name__ == '__main__':
    with open("../metadata/asr_annotated.csv", 'r') as f:
        df = pd.read_csv(f)
    rows_to_append = []
    for index, row in tqdm.tqdm(df.iterrows()):
        try:
            embedding = generate_embeddings(row["transcript"])
            rows_to_append.append(embedding)
        except Exception:
            rows_to_append.append(0)
    df.insert(2, "embedding", rows_to_append, True)

    # df["embedding"] = rows_to_append
    df.to_csv("../metadata/asr_annotated_embeddings.csv", index=False)
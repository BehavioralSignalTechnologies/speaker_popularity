# speaker_popularity
Automatically assessing public speakers popularity: A use-case for ted-talks

## Structure

* audio_features: folder containing code for extracting high-level aggregations of the behavioral posteriors
* metadata: folder containing all generated datasets
  * ted_main.csv: The original TED talks dataset
  * merged_metadata.csv: The above dataset merged with the transcripts
  * merged_metadata_popularity_std.csv: The dataset enriched with the target metrics (popularity + ratings)
  * merged_metadata_popularity_features_std.csv: The dataset enriched with target metrics and features used in classifiers
  * embeddings_transcript_clean.csv: The embeddings of the transcripts
* mfccs: The mfccs per file, calculated with librosa
* modeling_api_results_embeddings: The responses of our API for each file, containing the behavioral embeddings and posteriors
* results: The directory where experiments results are saved
* scripts: Various scripts that are used by the process (explained below)

## Install dependencies and environment

Make sure you have git lfs installed.
Clone the repo and create a virtual environment (python 3.8) and install the required dependencies

```shell
git clone https://github.com/BehavioralSignalTechnologies/speaker_popularity.git
cd speaker_popularity
virtualenv -p python3.8 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Unzip `modeling_api_results_embeddings/emb_part_1.zip` and `modeling_api_results_embeddings/emb_part_2.zip`
and place all files under the `modeling_api_results_embeddings` directory

## How to run:

1. Generate the dataset including the target labels (popularity + ratings). 
The generated dataset's name will be `metadata/merged_metadata_popularity_std.csv`

```shell
cd scripts
python3 generate_popularity.py
```

2. Enhance the dataset with the extracted features (behavioral embeddings + posteriors). 
The generated dataset's name will be `metadata/merged_metadata_popularity_features_std.csv`

```shell
python3 generate_dataset_with_aggregations.py
```

3. Generate the text embeddings of the transcripts by running the `clean_transcripts_embeddings` notebook. An OpenAI API key is required for this step.
The generated embeddings dataset will be `metadata/embeddings_transcript_clean.csv`

4. Run the experiment.

```shell
python3 run_experiment.py
```
The results wull be saved under the `results` folder. You can then run the cells in `features_analysis` notebook,
to visualize and explore the results (`results/scores.csv`)

## Additional scripts

* visualize_target_set: Contains visualization regarding the distributions of the popularity and ratings metrics
* eda: Various graphs and aggregations used as part of data exploration
* get_correlations: Prints the correlation matrix between target labels and features
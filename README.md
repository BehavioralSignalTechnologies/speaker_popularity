# speaker_popularity
Automatically assessing public speakers popularity: A use-case for ted-talks

## Structure

* audio_features: folder containing code for extracting high-level aggregations of the behavioral posteriors
* metadata: folder containing all generated datasets
* mfccs: The mfccs per file, calculated with librosa
* modeling_api_results_embeddings: The responses of our API for each file, containing the behavioral embeddings and posteriors
* results: The directory where experiments results are saved
* scripts: Various scripts that are used by the process (explained below)

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
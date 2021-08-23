# Mainrepo

Overview of the directories in this repository:

## analysis

Several python scripts for data set analysis. 
Some of the functions in these scripts are "one and done" functions that are not generalized.

## archiv

Archived evaluation output files containing the results presented in the thesis

- data_analysis: Numbers about the data set
- full_twitch: Results of running LSCD on the full Twitch data set with selected target words
- fuse_logs: Logs of running 50 epochs of fusing to gather Loss information
- fuse_results: synthetic evaluation results for fused models
- games_results: Results of synchronic/domain LSCD; 
  - dotalol_twostep_results: two-step approach for dota and lol
  - dota_cp_results: change-point approach for dota timeseries
- jeseme: Results of running change-point LSCD on the DTA timeseries data
- synth_eval: Total results for synthetic dataset experiments
- woc: results for WoC reproduction

## data

- emotes: Emote lists + Emote images
- freq_analysis: Data for the frequency graphs in Chapter "Change-point approach for selected words"
- targets: DURel and SURel target words as well as the list of selected words for Twitch

## docker

- Dockerfiles and requirements

## embeddings

- Embedding generation script and simple embedding query script

## evaluation

- Evaluation scripts
- evaluate_synthetic_data_results.py 

## kubernetes

- kubernetes config files and config creator scripts
- month_job_creator.py: 

## multimodal

- Python scripts for all steps of multimodal approach
- vocab.py: vocabulary generation
- embeddings.py: word embedding generation
- image_representations.py: using a pretrained CNN to compute image representations
- fuse.py: the actual fusing step
- twostep_lsc.py and changepoint_lsc.py: self-explanatory

## odenet

- Submodule: https://github.com/hdaSprachtechnologie/odenet
- German Wordnet with hypernyms and hyponyms
- Used for synthetic data set generation on german texts

## scripts

- Shell scripts used for several things. 
- The paths, commands etc in these scripts are hardcoded currently.

## semantic_change_detection

- Scripts for semantic change detection

## synthetic_data_generation

- Slightly adapted synthetic data generation from  https://github.com/alan-turing-institute/room2glo
- use_odenet.py is for using odenet instead of WordNet for german data

#  Experiments reproduction

For all of these script applications see also the "kubernetes" folder for concrete or template yaml config files

# Data and model locations on vingilot

- /home/stud/bernstetter/datasets
- /home/stud/bernstetter/models

## Data preprocessing

- Preprocess data individually depending on the data set
- All further scripts (except for multimodal) assume plain text files with one message/messageblock/sentence per line

## Synthetic Dataset generation

- directory: synthetic_data_generation/
- run get_freqs_and_wordnet_stats.py on corpus; results in a context_word_dict.json and a vocab_stats.csv
- run design_pseudowords.py; results in a pseudoword_dict.json
- run create_synthetic_dataset.py; create synthetic data set with custom time span. default is 2019-05 to 2020-04

## Embedding generation

- directory: embeddings/
- generate.py
- Will result in an output of "vec_128_w5_mc100_iter10_sg0/saved_model.gensim" (using the default parameters)

## Semantic Change Detection

- Adapted from: https://github.com/alan-turing-institute/room2glo/blob/master/semantic_change_detection_methods/
- directory: semantic_change_detection.py
- hamilton_semantic_change_measures.py = two-step approach. input are two gensim models; results in a cosine.tsv or neighborhood.tsv
- change_point_detection.py = change-point approach. 
  - input is a directory of time steps.
  - In this directory, the time steps will be sorted
  - the actual models need to be in directories of format "vec_{}_w{}_mc{}_iter{}_sg{}/saved_model.gensim"
  - several config options are available (which model to align to etc)
- both hamilton and change_point can have a "--targets=path/to/targets.csv" argument which is a list of target words
- hamilton_durel.py was used specifically for applying the two-step approach to 

## Evaluation

- directory: evaluation
- evaluate_synthetic_data_results.py: adapted from https://github.com/alan-turing-institute/room2glo/
  - input is a result file either from hamilton/two-step or change-point
  - Output are several results files, the most important one ending in _metrics.tsv
- get_ap_at_50.py takes such a metrics file as input and outputs exactly the AP@50
- the other scripts each compare two two-step/change-point result files to each other

## Multimodal

- build vocabulary with vocab.py
- train word embeddings with embeddings.py
- optionally get image_representations.py
- fuse.py
- twostep_lsc.py or changepoint_lsc.py
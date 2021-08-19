# Mainrepo

Overview of the directories in this repository:

## analysis

Several python scripts for data set analysis.

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




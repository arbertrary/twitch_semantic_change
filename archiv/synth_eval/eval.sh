#!/bin/bash

SCRIPT=/home/stud/bernstetter/ma/mainrepo/evaluation/evaluate_synthetic_data_results.py
DIR=/home/stud/bernstetter/ma/mainrepo/archiv/synth_eval/changepoint/
TW_PSEUDO=/home/stud/bernstetter/datasets/synthetic_twitch/pseudoword_dict.json
DTA_PSEUDO=/home/stud/bernstetter/datasets/dta/synth_dta/pseudoword_dict.json

for file in $DIR/tw*.tsv
do
	python $SCRIPT --results_dir=$DIR --results_fn=$(basename $file) --pseudoword_design_dict=$TW_PSEUDO --word_column=0
done

for file in $DIR/dta*.tsv
do
	python $SCRIPT --results_dir=$DIR --results_fn=$(basename $file) --pseudoword_design_dict=$DTA_PSEUDO --word_column=0
done

#!/bin/bash

BASEDIR=/home/stud/bernstetter/models/twitch_multimodal/synth_results/changepoint_15ep
FILENAME=time_series_analysis_standardized_output_f201905_l202004_alast_clast_s1000_v75

EVALDIR=/home/stud/bernstetter/ma/mainrepo/evaluation/
EVALSCRIPT=evaluate_synthetic_data_results.py
APAT50SCRIPT=get_ap_at_50.py

PSEUDOWORDS=/home/stud/bernstetter/datasets/twitch_multimodal/synthetic/pseudoword_dict.json

for dir in $BASEDIR/*/*/;
do
	echo "# ${dir/$BASEDIR/}";
	python3 $EVALDIR/$EVALSCRIPT --results_dir=$dir --results_fn=$FILENAME.tsv --word_column=0 --pseudoword_design_dict=$PSEUDOWORDS > /dev/null;
	echo "# AVG Prec at 50";
	python3 $EVALDIR/$APAT50SCRIPT --in_file=$dir/results/"$FILENAME"_metrics.tsv;
	printf "####\n\n";
done

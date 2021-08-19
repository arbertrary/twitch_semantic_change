#!/bin/bash

PREFIX="/home/armin/masterarbeit/repos/mainrepo/"

echo "$PREFIX"

python "$PREFIX"/preprocessing/preprocessing.py --infiles_rootdir="$PREFIX"/data/testdata/raw --outfiles_dir="$PREFIX"/data/testdata/working/ --ffz="$PREFIX"/data/emotes/2021/ffz_emotes.csv --bttv="$PREFIX"/data/emotes/2021/bttv_global_emotes.csv

python $PREFIX/preprocessing/create_corpusfiles.py --mode=f --infiles_rootdir=$PREFIX/data/testdata/working/ --outdir_path=$PREFIX/data/testdata/emote_filtered

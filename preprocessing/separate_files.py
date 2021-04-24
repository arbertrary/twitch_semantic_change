"""
Separate a chatlog csv file into files for each game
"""

import os
import pandas as pd
import argparse
import csv


def filterByGame(in_dir, out_dir, game_string):
    dataframes = []
    for f in os.listdir(in_dir):
        filename = os.path.join(in_dir, f)
        df = pd.read_csv(filename, sep=",")

        filtered = df.loc[df['game'] == game_string]

        dataframes.append(filtered)

    out_frame = pd.concat(dataframes)
    key_to_filename = str(game_string).replace(" ", "_").replace("/", "") + ".csv"
    out_filename = os.path.join(out_dir, key_to_filename)
    out_frame.to_csv(out_filename, header=True, index=False)


def separate(in_dir, out_dir, separator):
    for f in os.listdir(in_dir):
        filename = os.path.join(in_dir, f)
        df = pd.read_csv(filename, sep=",")

        grouped_by = df.groupby(separator)

        for key, item in grouped_by:
            key_to_filename = str(key).replace(" ", "_").replace("/", "") + ".csv"
            out_filename = os.path.join(out_dir, key_to_filename)

            if os.path.exists(out_filename):
                item.to_csv(out_filename, mode='a', header=False, index=False)
            else:
                item.to_csv(out_filename, mode='w', header=True, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_dir", type=str, help="Directory where the textfiles are located")
    parser.add_argument("-o", "--outfiles_dir", type=str, help="Directory where the files should be written")
    parser.add_argument("-sep", "--separate_by", type=str, help="by which column to separate/groupby")
    parser.add_argument("-f", "--filter_game", type=str,
                        help="if this flag is set, files are only filtered by that one game")
    options = parser.parse_args()

    indir = options.infiles_dir
    outdir = options.outfiles_dir

    # indir = "../testdata2/clean_en_sorted/"
    # outdir = "../testdata2/filtertest/"

    os.makedirs(outdir, exist_ok=True)

    if options.filter_game:
        filterByGame(indir, outdir, options.filter_game)
    else:
        if options.separate_by == "game":
            separate(indir, outdir, "game")
        elif options.separate_by == "chid":
            separate(indir, outdir, "chid")
        else:
            raise ValueError("separator doesn't exist")

import argparse
import datetime
import os
import multiprocessing
import json
import numpy as np
import pandas as pd
import gzip
import sys
import csv


# with gzip.open("file.gz", "r") as in_file:
#     reader = csv.DictReader(in_file, delimiter=",")

def maybe_return_pseudoword(context_word, year_month_index):
    if context_word not in context_words:
        return context_word


    # look up which pseudoword the current context_word is associated with.
    pseudoword = context_words[context_word]['pseudoword']

    #  look up which schema the associated pseudoword belongs to.
    pseudoword_type = context_words[context_word]['pseudoword_type']

    # look up which 'set' the context_word belongs to (i.e. what 'kind' of context_word it is)
    # e.g. Schema C2 pseudowords have one or more context_words which gradually decrease in frequency over
    # time (set 1), and one or more context_words which gradually increase in frequency over time (set 2).
    set_number = context_words[context_word]['set_number']

    # Schema C3 pseudowords and Schema D4 pseudowords each have a set of context_words whose replacement probabilities at
    # a given time-step are specified by a multinomial distibution drawn with a Dirichlet prior.
    if pseudoword_type == 'D4' or (pseudoword_type == 'C3' and set_number == 1):

        # So, if the current context_word is associated with a C3 or D4 pseudoword, we need to look up which index in
        # the multinomial distribution corresponds to it.
        dist_index = context_words[context_word]['dist_index']

        # We then retrieve the pseudoword-insertion-probability, given the current context_word's set number, the
        # current time-step, and the current context_word's index in the current time-step's multinomial distribution.
        p_insert = pseudoword_insert_probs[pseudoword]["p{}_array_series".format(set_number)][year_month_index][
            dist_index]

    else:
        try:
            # retrieve the pseudoword-insertion-probability given the current context_word's set number, and the
            # current time-step.
            p_insert = pseudoword_insert_probs[pseudoword]["p{}_series".format(set_number)][year_month_index]

        except TypeError:
            print(pseudoword)
            print(context_word)
            print(pseudoword_insert_probs[pseudoword])
            raise ()

    # set 'insert' to 1 with probability p_insert, or to 0 with probability 1-p_insert
    insert = np.random.RandomState().choice(2, 1, p=[1 - p_insert, p_insert])

    # if 'insert' was set to 1, we will replace the current context_word with its associated pseudoword
    if insert:
        return pseudoword

    #  if 'insert' was set to 0, we will not replace the current context_word.
    else:
        return context_word


def synthesize_message(message: str, year_month_index):
    try:
        msg = message.split()
    except AttributeError as ae:
        return message

    msg_out = [maybe_return_pseudoword(w, year_month_index) for w in msg]
    # msg_out = []
    # for i in range(len(msg)):
    #     if msg[i] in context_words:
    #         pseudoword = maybe_return_pseudoword(msg[i], year_month_index)
    #         if pseudoword:
    #             msg_out.append(pseudoword)
    #         else:
    #             msg_out.append(msg[i])
    #     else:
    #         msg_out.append(msg[i])

    return " ".join(msg_out)


def create_synthetic(year_month_index, subsampling, subsample_percent: float):
    year_month = year_months[year_month_index]

    # with gzip.open(os.path.join(outfiles_rootdir, year_month[:4], year_month), "wt") as outfile:

    df = pd.read_csv(input_filepath,names=["msg","emotenames"],sep="\t")
    if subsampling:
        df = df.sample(frac=subsample_percent)

    df["msg"] = df["msg"].apply(lambda x: synthesize_message(x, year_month_index))

    df.to_csv(os.path.join(outfiles_rootdir, year_month[:4], year_month),sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # python3 create_synthetic_dataset.py --i "../testdata/synthetic/10p_sample_201911/10p_201911031640.csv.gz" -o "../testdata/synthetic/" -c "../testdata/synthetic/context_word_dict.json" -p "../testdata/synthetic/pseudoword_dict.json"
    parser.add_argument("-i", "--input_filepath", type=str,
                        default='/home/stud/bernstetter/datasets/synthetic_twitch/concat_10p_201911.csv.gz',
                        help="path to file in which data for the month we want to use is stored")
    parser.add_argument("-o", "--outfiles_rootdir", type=str,
                        default='/home/stud/bernstetter/datasets/synthetic_twitch/',
                        help="path to directory where synthetic dataset should be written")
    parser.add_argument("-c", "--context_word_dict_filepath", type=str,
                        default='/home/stud/bernstetter/datasets/synthetic_twitch/context_word_dict.json',
                        help="path to file in which context word dict is stored")
    parser.add_argument("-p", "--pseudoword_dict_filepath", type=str,
                        default='/home/stud/bernstetter/datasets/synthetic_twitch/pseudoword_dict.json',
                        help="path to file in which pseudoword dict is stored")

    parser.add_argument("-s", "--subsampling", type=int, default=1,
                        help="whether or not to subsample from original data. 1 = yes, 0 = no.")
    parser.add_argument("-sp", "--subsampling_percent", type=int, default=70,
                        help="size of sample (percent of original data) e.g. 70")

    parser.add_argument("-sy", "--start_year", type=int, default=2019, help="start year: integer, e.g. 2012")
    parser.add_argument("-sm", "--start_month", type=int, default=5, help="start month: integer, e.g. 6")
    parser.add_argument("-ey", "--end_year", type=int, default=2020, help="end year: integer, e.g. 2014")
    parser.add_argument("-em", "--end_month", type=int, default=4, help="end month: integer, e.g. 4")

    options = parser.parse_args()

    subsampling = options.subsampling
    subsampling_percent = options.subsampling_percent

    if subsampling:
        options.outfiles_rootdir += '/subsampled_{}/'.format(subsampling_percent)

    input_filepath = options.input_filepath
    outfiles_rootdir = options.outfiles_rootdir
    context_word_dict_filepath = options.context_word_dict_filepath
    pseudoword_dict_filepath = options.pseudoword_dict_filepath
    start_year = options.start_year
    end_year = options.end_year
    start_month = options.start_month
    end_month = options.end_month

    year_months = []

    for year in range(start_year, end_year + 1):

        os.makedirs(os.path.join(outfiles_rootdir, str(year)), exist_ok=True)

        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            elif year == end_year and month > end_month:
                break

            year_months.append("{}{:02}.txt".format(year, month))

    n_timesteps = len(year_months)

    # load the pseudoword design dictionaries that were previously created using design_pseudowords.py.
    with open(context_word_dict_filepath, 'r') as infile:
        context_words = json.load(infile)

    with open(pseudoword_dict_filepath, 'r') as infile:
        pseudoword_insert_probs = json.load(infile)

    for year_month_index in range(n_timesteps):  # i.e. for each timestep...
        subsampling_percent = float(0.01 * 70)
        create_synthetic(year_month_index, subsampling, subsampling_percent)

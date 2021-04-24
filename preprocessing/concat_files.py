#!/usr/bin/python -u

import glob
import os
import gzip
import itertools

import numpy as np
import os
import pandas as pd
import argparse
import time
import logging


class ChatYielder():
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            file = os.path.join(self.dirname, fname)
            df = pd.read_csv(file, sep=",")
            df.style.hide_index()

            grouped_df = df.groupby("chid")

            # pd.set_option('display.max_rows', 10)

            # grouped_df.describe()
            for key, item in grouped_df:
                ts = item["ts"]

                # sent = list(itertools.chain.from_iterable([str(row).split() for row in item["msg"]]))
                # yield sent

                ts_range = np.arange(ts.min() - 30000, ts.max() + 30000, 30000)
                test = item.groupby(pd.cut(item["ts"], ts_range))
                # print(len(test))
                for k, i in test:
                    if not i.empty:
                        sent = " ".join([str(row) for row in i["msg"]]) + "\n"

                        yield sent


def createCorpusFile(in_path, out_path):
    start_time = time.time()
    logging.info("# STARTED CREATING CORPUS FILE")
    print(out_path)
    df = pd.read_csv(in_path, compression="gzip", sep=",")
    df.style.hide_index()

    df.sort_values(by=["chid", "ts"])

    logging.info("# SORTED at %s seconds" % (time.time() - start_time))

    grouped_df = df.groupby("chid")
    logging.info("# grouped by CHID at %s seconds" % (time.time() - start_time))

    group_list = []

    with open(out_path, "w") as outfile:
        for key, item in grouped_df:
            ts = item["ts"]

            ts_time = time.time()
            logging.info("\t # BEFORE timestamp GROUPING at %s" % ts_time)

            ts_range = np.arange(ts.min() - 30000, ts.max() + 30000, 30000)
            test = item.groupby(pd.cut(item["ts"], ts_range))
            group_list = [(index, group) for index, group in test if len(group) > 0]
            logging.info("\t # TIMESTAMP GROUPING took %s seconds" % (time.time() - ts_time))

            # test = df.replace("", nan_value, inplace=True)
            # test = test.dropna(subset=["msg"])

            # df.dropna(subset, inplace=True)
            iter_time = time.time()
            logging.info("\t # BEFORE ITERATING THE GROUPS at %s" % iter_time)

            for k, i in group_list:
                # print("# group")
                # print(k)
                # print("\n# ITEM")
                # print(i)
                # print("############################\n")
                sent = " ".join([str(row) for row in i["msg"]]) + "\n"
                outfile.write(sent)
            logging.info("# ITERATING AND writing 30s blocks took %s seconds" % (time.time() - iter_time))
    logging.info("--- %s seconds IN TOTAL ---" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str)
    parser.add_argument("-o", "--outfile_path")
    logging.basicConfig(filename="/home/stud/bernstetter/datasets/synthetic_twitch/grouped_synth/corpus.log",
                        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    # logging.basicConfig(filename="corpus.log",
    #                     format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    options = parser.parse_args()

    if os.path.isfile(options.infiles_rootdir):
        createCorpusFile(options.infiles_rootdir, options.outfile_path)
    else:
        # dirpath = "/home/stud/bernstetter/datasets/twitch_sampled/5Kx5K/201905/"
        chat = ChatYielder(options.infiles_rootdir)
        with open(options.outfile_path, "w+") as outfile:
            outfile.writelines(chat)
    # out = "/home/stud/bernstetter/datasets/twitch_sampled/5Kx5K/201905_corpus.txt"

    # if __name__ == '__main__':
    #     interesting_files = glob.glob("/home/stud/bernstetter/datasets/synthetic_twitch/10p_sample_201911/*.txt")
    #     df = pd.concat((pd.read_csv(f, header=0) for f in interesting_files))
    #     df.to_csv("/home/stud/bernstetter/datasets/synthetic_twitch/concat_10p_201911.csv")
    #
    #     with open("/home/stud/bernstetter/datasets/synthetic_twitch/concat_10p_201911.csv", "rb") as f_in:
    #         with gzip.open("/home/stud/bernstetter/datasets/synthetic_twitch/concat_10p_201911.csv.gz", "w") as f_out:
    #             f_out.writelines()

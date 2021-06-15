import argparse
import logging
import time
import pandas as pd
import os
import multiprocessing
import numpy as np
import re


def filter_non_emote_messages(in_filepath):
    df = pd.read_csv(in_filepath, sep=",")
    df.dropna(subset=["emotes", "extemotes"], how="all", inplace=True)

    filename = "filtered_" + os.path.basename(in_filepath)

    out_filepath = os.path.join(out_path, filename)
    df.to_csv(out_filepath, index=False)


def filter_non_emote_messages_to_tsv(in_filepath):
    df = pd.read_csv(in_filepath, sep=",")
    df.dropna(subset=["emotes", "extemotes"], how="all", inplace=True)

    filename = "filtered_" + os.path.basename(in_filepath).replace(".csv", ".tsv")
    df["emotenames"] = df.apply(
        lambda row: get_emotes_in_msg_for_df(row["msg"],
                                             str(row["emotes"]) + "/" + str(row["extemotes"])), axis=1)
    out_filepath = os.path.join(out_path, filename)

    df[["msg", "emotenames"]].to_csv(out_filepath, index=False, sep="\t")


def emote_corpusfile(in_filepath):
    df = pd.read_csv(in_filepath, sep=",")
    filename = os.path.basename(in_filepath)

    df.dropna(subset=["emotes", "extemotes"], how="all", inplace=True)

    grouped_df = df.groupby("chid")
    with open(os.path.join(out_path, "emotes_" + filename), "w") as outfile:
        for key, item in grouped_df:
            ts = item["ts"]

            ts_range = np.arange(ts.min() - 30000, ts.max() + 30000, 30000)
            test = item.groupby(pd.cut(item["ts"], ts_range))
            group_list = [(index, group) for index, group in test if len(group) > 0]
            for k, i in group_list:
                i["tempemotes"] = i["emotes"].astype(str) + "/" + i["extemotes"].astype(str)

                # gets the actual emote names. more computationally intensive
                emotes = i["tempemotes"].values.tolist()
                messages = i["msg"].values.tolist()
                zipped = zip(messages, emotes)
                emote_messages = [get_emotes_in_msg_for_df(str(msg), str(em)) for (msg, em) in zipped]

                # initial try: gets the emote ids
                # emote_messages = [emote_sentence(row) for row in i["tempemotes"]]
                sent = " ".join(emote_messages) + "\n"
                outfile.write(sent)


def get_emotes_in_msg_for_df(message: str, emote_ranges):
    emotes = emote_ranges.split("/")

    emote_indices = [x.split(":")[1].strip() for x in emotes if x != "nan"]
    # emote_indices format: ["5-7,8-12", "4-5"]
    first_occ = [x.split(",")[0].strip() for x in emote_indices]
    # first_occ format: ["5-7", "4-5"]
    pattern = re.compile(r"(\d+)-(\d+)")

    emotenames = []
    for ind in first_occ:
        m = re.match(pattern, ind)
        i = m.group(1)
        j = m.group(2)

        emote = message[int(i):int(j) + 1]
        emotenames.append(emote)

    return " ".join(emotenames)


def ungrouped_corpusfile(in_filepath):
    df = pd.read_csv(in_filepath, sep=",", error_bad_lines=False)
    filename = os.path.basename(in_filepath).replace(".csv", ".txt")

    df = df["msg"]

    df.to_csv(os.path.join(out_path, "ungrouped_" + filename), index=None, header=None)


def corpusfile(in_filepath):
    df = pd.read_csv(in_filepath, sep=",", error_bad_lines=False)
    df.style.hide_index()

    filename = os.path.basename(in_filepath).replace(".csv", ".txt")

    grouped_df = df.groupby("chid")
    with open(os.path.join(out_path, "grouped_" + filename), "w") as outfile:
        for key, item in grouped_df:
            # sentence = " ".join([str(row) for row in item["msg"]]) + "\n"
            # outfile.write(sentence)

            ts = item["ts"]

            ts_range = np.arange(ts.min() - 30000, ts.max() + 30000, 30000)
            test = item.groupby(pd.cut(item["ts"], ts_range))
            group_list = [(index, group) for index, group in test if len(group) > 0]
            # print(len(test))
            for k, i in group_list:
                sent = " ".join([str(row) for row in i["msg"]]) + "\n"
                outfile.write(sent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str)
    parser.add_argument("-o", "--outdir_path")
    parser.add_argument("-m", "--mode", type=str, default="g")

    options = parser.parse_args()

    in_path = options.infiles_rootdir
    out_path = options.outdir_path
    os.makedirs(out_path, exist_ok=True)

    pool = multiprocessing.Pool(11)
    filelist = [os.path.join(in_path, file) for file in os.listdir(in_path)]

    if options.mode == "g":
        pool.map(corpusfile, filelist)
    elif options.mode == "u":
        pool.map(ungrouped_corpusfile, filelist)
    elif options.mode == "f":
        pool.map(filter_non_emote_messages_to_tsv, filelist)
    else:
        pool.map(emote_corpusfile, filelist)

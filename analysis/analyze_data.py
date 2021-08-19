import json
import os
from collections import Counter, OrderedDict
import argparse
import itertools
import csv
import pandas as pd
import numpy as np
import math
from nltk.corpus import stopwords


def analyze_sorted(file: str):
    df = pd.read_csv(file, sep=",")
    df.style.hide_index()

    grouped_df = df.groupby("chid")

    pd.set_option('display.max_rows', 10)

    # grouped_df.describe()
    for key, item in grouped_df:
        ts = item["ts"]

        sent = list(itertools.chain.from_iterable([row.split() for row in item["msg"]]))

        print(sent)

        # ts_range = np.arange(ts.iloc[0]-30000, ts.iloc[-1]+30000, 30000)
        # test = item.groupby(pd.cut(item["ts"], ts_range))
        # print(len(test))
        # for k, i in test:
        #     if not i.empty:
        #         print(i[["ts", "msg"]], "\n\n")
        #         # print(pd.concat(i["msg"]))
        #
        #         sent = list(itertools.chain.from_iterable([row.split() for row in i["msg"]]))
        #
        #         print(sent)

        break


def analyze_emotes(csv_rootdir, out_dir):
    # the counter for "occurrence of emotes, regardless of how often in a message it is repeated"
    emotecounter = Counter()
    total_emotecounter = Counter()
    channelemote_counter = {}

    for file in os.listdir(csv_rootdir):
        if file.endswith(".gz"):
            continue
        filepath = os.path.join(csv_rootdir, file)
        with open(filepath, "rt") as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter=",")

            for line in csv_reader:
                channel = line["chid"]

                em_len = len(line["emotes"])
                ext_len = len(line["extemotes"])
                if em_len == 0 and ext_len == 0:
                    continue
                elif em_len == 0:
                    combined_emotes = line["extemotes"]
                elif ext_len == 0:
                    combined_emotes = line["emotes"]
                else:
                    combined_emotes = line["emotes"] + "/" + line["extemotes"]

                emotes = [x.split(":") for x in combined_emotes.split("/")]
                for e in emotes:
                    emote = e[0]
                    emotecounter.update([emote])
                    repetitions = e[1].split(",")

                    occurrences = len(repetitions)
                    total_emotecounter.update({emote: occurrences})

                    if channel in channelemote_counter:
                        ch = channelemote_counter[channel]
                        if emote in ch:
                            channelemote_counter[channel][emote]["msgs"] += 1
                            channelemote_counter[channel][emote]["total"] += occurrences
                        else:
                            channelemote_counter[channel][emote] = {"msgs": 1, "total": occurrences}
                    else:
                        channelemote_counter[channel] = {emote: {"msgs": 1, "total": occurrences}}

    with open(os.path.join(out_dir, "emotecounter.json"), "w") as outfile:
        json.dump(OrderedDict(emotecounter.most_common()), outfile, indent=2)

    with open(os.path.join(out_dir, "total_emotecounter.json"), "w") as outfile:
        json.dump(OrderedDict(total_emotecounter.most_common()), outfile, indent=2)

    with open(os.path.join(out_dir, "channel_emotecounter.json"), "w") as outfile:
        json.dump(channelemote_counter, outfile, indent=2)


def combine_dict():
    combined_dict = {}
    for month in os.listdir("channeljson"):

        print(month)
        with open(os.path.join("channeljson", month), "r") as infile:
            channeldata = json.loads(infile.read())
            for key in channeldata:
                if key in combined_dict:
                    combined_dict[key]["vcnt_summed"] += channeldata[key]["vcnt_summed"]
                    combined_dict[key]["n_msgs"] += channeldata[key]["n_msgs"]
                else:
                    combined_dict[key] = channeldata[key]

    with open("channeljson/combined_channels.json", "w") as outfile:
        json.dump(combined_dict, outfile)


def count_messages():
    for month in os.listdir("channeljson"):
        n_msgs = 0
        with open(os.path.join("channeljson", month), "r") as infile:
            channeldata = json.loads(infile.read())

            for key in channeldata:
                n_msgs += channeldata[key]["n_msgs"]
        print("# " + month + ": " + str(n_msgs))


def avg_viewers():
    avg_viewers_dict = {}
    with open(os.path.join("channeljson", "combined_channels.json"), "r") as infile:
        channeldata = json.loads(infile.read())
        for key in channeldata:
            avg_viewers_dict[key] = {"ch": channeldata[key]["ch"],
                                     "avg_viewers": channeldata[key]["vcnt_summed"] / channeldata[key]["n_msgs"]}

    with open("channeljson/channels_vcnt.json", "w") as outfile:
        for key in avg_viewers_dict:
            outfile.write("{\"" + key + "\":" + str(avg_viewers_dict[key]) + "}\n")


def analyze_users():
    unique_users = {}
    twitch_dataset = "/home/stud/bernstetter/datasets/twitch/"
    for month in os.listdir(twitch_dataset):
        print(month)
        if os.path.isfile(os.path.join(twitch_dataset, month)):
            continue
        else:
            user_json = os.path.join(twitch_dataset, month, "users.json")
            with open(user_json) as infile:
                users = json.loads(infile.read())
                unique_users.update(users)

    print(len(unique_users))

    with open(os.path.join(twitch_dataset, "unique_users.json"), "w") as outfile:
        json.dump(unique_users, outfile)


def analyze_days():
    twitch_dataset = "/home/stud/bernstetter/datasets/twitch/"
    months_sorted = sorted(os.listdir(twitch_dataset))
    for month in months_sorted:
        print("# " + month + ":")
        files_per_day = {}
        if os.path.isfile(os.path.join(twitch_dataset, month)):
            continue
        else:
            path = os.path.join(twitch_dataset, month, "clean")
            for file in os.listdir(path):
                day = file[6:8]
                if day in files_per_day:
                    files_per_day[day] += 1
                else:
                    files_per_day[day] = 1

            keyset_sorted = sorted(files_per_day.keys())
            print("# Recorded days: " + str(len(keyset_sorted)))
            for key in keyset_sorted:
                print(key + ": " + str(files_per_day[key]))
        print("")


def wc_median(in_dir):
    msg_lengths = []
    for file in os.listdir(in_dir):
        filename = os.path.join(in_dir, file)
        with open(filename, "r") as f:
            for line in f.readlines():
                msg_lengths.append(len(line.split()))
    
    c = Counter()
    c.update(msg_lengths)
    avg = np.mean(msg_lengths)
    median = np.median(msg_lengths)
    print("Average message length:")
    print(avg)
    print("Median message length:")
    print(median)
    print(c)

def unique_emotes(in_dir):
    c = Counter()

    for file in os.listdir(in_dir):
        print(file)
        filepath = os.path.join(in_dir, file)
        with open(filepath, "r") as emotefile:
            for line in emotefile.readlines():
                c.update(line.split())

    emotes = [e for e in c if c[e] >= 500]
    print(len(emotes))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str,
                        default="/home/stud/bernstetter/datasets/twitch_sampled/5Kx5K/analysis/month_freqs/",
                        help="path to directory where models are stored")
    parser.add_argument("-o", "--outfiles_dir", type=str,
                        default="/home/stud/bernstetter/ma/initial/testdata2/")
    parser.add_argument("-m", "--month", type=str, default="2019-05")

    options = parser.parse_args()
    in_path = options.infiles_rootdir
    out_path = options.outfiles_dir
    #wc_median(in_path)
    unique_emotes(in_path)

    # csvfiles_path = os.path.join(options.infiles_rootdir, options.month, "clean")
    # out_path = os.path.join(options.outfiles_dir, options.month, "analysis")
    # os.makedirs(out_path, exist_ok=True)
    # analyze_emotes(csvfiles_path, out_path)

    # for file in os.listdir(csvfiles_path):
    #     filepath = os.path.join(csvfiles_path, file)
    #     analyze_sorted(filepath)
    # break

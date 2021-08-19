import json
import os
from collections import Counter, OrderedDict
import csv
import pandas as pd
import math
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from itertools import cycle

# matplotlib.use('TkAgg')

def temp_get_wordcounts(in_dir, out_dir):
    for file in os.listdir(in_dir):
        infile_path = os.path.join(in_dir, file)
        print(infile_path)
        out_path = os.path.join(out_dir, "pseudoword_counts_" + file + ".json")
        c = Counter()
        with open(infile_path, "r") as infile:
            print("# file: " + infile_path)
            for line in infile:
                c.update(line.strip().split())
        c = {x: count for x, count in sorted(c.items(), key=lambda item: item[0]) if count >= 100 and "pseudoword" in x}
        with open(out_path, "w") as outfile:
            json.dump(OrderedDict(c), outfile, indent=2)


def get_wordcounts(in_dir, out_dir, month):
    c = Counter()
    out_path = os.path.join(out_dir, "counts_" + month + ".json")

    for file in os.listdir(in_dir):
        infile_path = os.path.join(in_dir, file)

        if not os.path.isfile(infile_path):
            continue

        with open(infile_path, "r") as infile:
            print("# file: " + infile_path)
            for line in infile:
                c.update(line.strip().split())

    c = {x: count for x, count in c.items() if count >= 100}
    with open(out_path, "w") as outfile:
        json.dump(OrderedDict(c), outfile, indent=2)


def get_wordfreqs(in_dir, out_dir):
    """
    :param in_dir: Directory where all of the Counter word occurrence json files lie
    :return:
    """

    vocab = set()
    month_dicts = []
    for filename in sorted(os.listdir(in_dir)):
        file_path = os.path.join(in_dir, filename)

        with open(file_path, "r") as infile:
            data = infile.read()
            json_data = json.loads(data)
            new_dict = {}
            total = sum(json_data.values())

            for key in json_data:
                if json_data[key] < 100:
                    continue
                vocab.add(key)
                new_dict[key] = math.log(json_data[key] / total)

            month_dicts.append(new_dict)

    with open(os.path.join(out_dir, "wc100_logfrequencies.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            ["word", "201905", "201906", "201907", "201908", "201909", "201910", "201911", "201912", "202001", "202002",
             "202003", "202004"])

        for word in vocab:
            wordfreqs = [word]
            write = True
            for month in month_dicts:
                if not month.get(word):
                    write = False
                    break
                else:
                    wordfreqs.append(month[word])
                    write = True
            if write:
                writer.writerow(wordfreqs)


def analyze_frequencies(frequency_file):
    df = pd.read_csv(frequency_file)
    diffs = []
    todrop = []
    for index, row in df.iterrows():
        if str(row[0]).lower() in stopwords.words("english"):
            todrop.append(index)
        temp = list(row[1:])
        maxdiff = abs(max(temp) - min(temp))

        diffs.append(maxdiff)

    df["MaxDiff"] = diffs
    df = df.drop(todrop)
    df = df.sort_values("MaxDiff", ascending=False)

    df.to_csv("../data/freq_analysis/wc100_maxdiffs_log.csv", index=False)


def plot_frequencies(diffs_file, words):
    df = pd.read_csv(diffs_file)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    y = df.columns[1:-1]

    lines = ["-", "--", "-.", ":"]

    linecycler = cycle(lines)

    for w in words:
        temp = df.loc[df['word'] == w]
        if temp.empty:
            continue
        values = temp.values[0]
        print(values[1:-1])
        ax.plot(y, values[1:-1], next(linecycler), label=w)  # Plot some data on the axes.

    fig.set_size_inches(10, 5)
    ax.set_xlabel('Months')  # Add an x-label to the axes.
    ax.set_ylabel('log frequency')  # Add a y-label to the axes.
    ax.set_title("Log Frequency of selected words related to Covid-19")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.show()


def plot_googletrends(csvfile):
    df = pd.read_csv(csvfile, index_col=0, parse_dates=True)
    df.plot(figsize=(10, 5.4), style=["-", "--", "-.", ":"])
    plt.title("Google Trends of words related to Covid-19")
    plt.xlabel("Months")
    plt.ylabel("Interest over time")

    plt.show()


if __name__ == '__main__':
    freqs = "../data/freq_analysis/wc100_logfrequencies.csv"
    diffs = "../data/freq_analysis/wc100_maxdiffs_log.csv"

    # words = ["simp", "boomer", "mald", "KEKW"]
    # words = ["hongkong", "blizzard", "revolution", "china", "hong", "kong", "censorship"]
    # words = ["PogChamp", "Pog", "Poggers", "EleGiggle", "TriHard"]
    words = ["corona", "virus", "lockdown", "quarantine"]

    plot_frequencies(diffs, words)

import json
import os
from collections import Counter, OrderedDict
import csv
import pandas as pd
import math
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle


# matplotlib.use('TkAgg')


def get_wordcounts(in_dir, out_dir):
    for file in os.listdir(in_dir):
        c = Counter()
        infile_path = os.path.join(in_dir, file)

        if not os.path.isfile(infile_path):
            continue

        with open(infile_path, "r") as infile:
            print("# file: " + infile_path)
            for line in infile:
                c.update(line.strip().split())

        out_path = os.path.join(out_dir, "freq_" + file)

        with open(out_path, "w") as outfile:
            json.dump(OrderedDict(c.most_common()), outfile, indent=2)


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
            json_data = json.load(infile)
            new_dict = {}
            total = sum(json_data.values())

            for key in json_data:
                if json_data[key] < 20:
                    continue
                vocab.add(key)
                new_dict[key] = math.log(json_data[key] / total)

            month_dicts.append(new_dict)

    with open(os.path.join(out_dir, "wc100_frequencies.csv"), "w") as csvfile:
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
                    # wordfreqs.append(0.0)
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
        # print(index)
        temp = list(row[1:])
        # maxdiff = abs(math.log(max(temp)) - math.log(min(temp)))
        maxdiff = abs(max(temp) - min(temp))

        diffs.append(maxdiff)
        # print(temp)

    df["MaxDiff"] = diffs
    df = df.drop(todrop)
    df = df.sort_values("MaxDiff", ascending=False)

    df.to_csv("../testdata2/wc20_maxdiffs_log.csv", index=False)
    # print(df.head(20))


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
    ax.set_title("Log Frequency of Words related to the Blitzchung Controversy")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.savefig("covid_freqs.png")
    plt.show()


if __name__ == '__main__':
    # freqs = "../testdata2/wc100_logfrequencies.csv"
    # freqs = "../testdata2/wc20_logfrequencies.csv"
    # analyze_frequencies(freqs)

    # diffs = "../testdata2/maxdiffs_log.csv"
    diffs = "../testdata2/wc20_maxdiffs_log.csv"
    # words = ["SIMP", "simp", "boomer", "ninja"]
    # words =["hongkong", "blizzard", "revolution", "china","hong", "kong"]
    # words = ["PogChamp", "Pog", "Poggers", "EleGiggle", "TriHard"]
    words = ["rona", "lockdown", "corona", "quarantine", "virus"]
    # The emote YEP was submitted to FrankerFaceZ[2] for the first time on December 15th, 2019.
    # words = ["KEY", "YEP", "hoursBrotherhood", "!drop", "PauseChamp"]

    plot_frequencies(diffs, words)



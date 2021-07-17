import os
import re
from csv import DictReader
from collections import Counter, OrderedDict
from itertools import chain
import argparse
import json


def build_vocabulary2(in_path: str, out_path: str, min_count: int, skip_emotes: int):
    vocab = {}
    filepath = in_path
    with open(filepath, "r", encoding="utf-8") as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        for row in reader:
            message = row["msg"].strip().split()
            emotenames = row["emotenames"].split()

            for word in message:
                if skip_emotes == 1 and word in emotenames:
                    continue
                if word not in vocab:
                    vocab[word] = {"emotes": Counter(emotenames), "count": 1}
                else:
                    vocab[word]["emotes"].update(emotenames)
                    vocab[word]["count"] += 1

    c = {x: {"emotes": {k:dct["emotes"][k] for k in dct["emotes"] if dct["emotes"][k] >= min_count}, "count": dct["count"]} for x, dct in
         sorted(vocab.items(), key=lambda y: y[1]["count"], reverse=True) if
         dct["count"] >= min_count}

    with open(out_path, "w") as outfile:
        json.dump(OrderedDict(c), outfile, indent=2)


def build_vocabulary(in_path: str, out_path: str, min_count: int, skip_emotes: int):
    counter = Counter()

    # for file in os.listdir(in_path):
    #    filepath = os.path.join(in_path, file)
    filepath = in_path
    with open(filepath, "r", encoding="utf-8") as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        for row in reader:
            message = row["msg"].strip().split()
            emotenames = row["emotenames"].split()

            tuplelist = []
            for word in message:
                if skip_emotes == 1 and word in emotenames:
                    continue
                tpl = "|".join([word] + emotenames)
                tuplelist.append(tpl)

            print(tuplelist)
            counter.update(tuplelist)

    c = {x: count for x, count in sorted(counter.items(), key=lambda y: y[1], reverse=True) if count >= min_count}

    with open(out_path, "w") as outfile:
        json.dump(OrderedDict(c), outfile, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str)
    parser.add_argument("-o", "--outdir_path")
    parser.add_argument("-m", "--min", type=int, default=100)
    parser.add_argument("-e", "--skip_emotes", type=int, default=0)
    args = parser.parse_args()

    indir = args.infiles_rootdir
    outdir = args.outdir_path
    os.makedirs(os.path.dirname(outdir), exist_ok=True)

    build_vocabulary(indir, outdir, args.min, args.skip_emotes)
    # build_vocabulary2("../data/testdata/emote_filtered/filtered_201911031555.txt", "vocab.json", min_count=2,
    #                   skip_emotes=1)

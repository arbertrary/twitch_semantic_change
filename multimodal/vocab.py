import os
import re
from csv import DictReader
from collections import Counter, OrderedDict
from itertools import chain
import argparse
import json


def build_global_vocabulary(in_path: str, out_path: str, min_count: int, emote_min_count:int, skip_emotes: int):
    vocab = {}

    if os.path.isfile(in_path):
        filepaths = [in_path]
    elif os.path.isdir(in_path):
        filepaths = [os.path.join(in_path, d) for d in os.listdir(in_path)]
    else:
        raise ValueError

    for file in filepaths:
        with open(file, "r", encoding="utf-8") as tsvfile:
            reader = DictReader(tsvfile, delimiter="\t")
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

    c = {x: {"emotes": {k: dct["emotes"][k] for k in dct["emotes"] if dct["emotes"][k] >= emote_min_count},
             "count": dct["count"]} for x, dct in
         sorted(vocab.items(), key=lambda y: y[1]["count"], reverse=True) if
         dct["count"] >= min_count}

    with open(out_path, "w") as outfile:
        json.dump(OrderedDict(c), outfile, indent=2)


def build_local_vocabulary(in_path: str, out_path: str, min_count: int, skip_emotes: int):
    counter = Counter()
    if os.path.isfile(in_path):
        filepaths = [in_path]
    elif os.path.isdir(in_path):
        filepaths = [os.path.join(in_path, d) for d in os.listdir(in_path)]
    else:
        raise ValueError

    for file in filepaths:
        with open(file, "r", encoding="utf-8") as tsvfile:
            reader = DictReader(tsvfile, delimiter="\t")
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
    parser.add_argument("-em", "--em_min", type=int, default=100)
    parser.add_argument("-e", "--skip_emotes", type=int, default=0)
    parser.add_argument("--local_vocab", action="store_true", default=False)
    args = parser.parse_args()

    indir = args.infiles_rootdir
    outdir = args.outdir_path
    os.makedirs(os.path.dirname(outdir), exist_ok=True)

    if args.local_vocab:
        build_local_vocabulary(indir, outdir, args.min, args.skip_emotes)
    else:
        build_global_vocabulary(indir, outdir, args.min,args.em_min, args.skip_emotes)

    # build_vocabulary2("../data/testdata/emote_filtered/filtered_201911031555.txt", "vocab.json", min_count=2,
    #                   skip_emotes=1)

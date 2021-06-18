import os
import re
from csv import DictReader
from collections import Counter, OrderedDict
from itertools import chain
import argparse
import json


def build_vocabulary(in_path: str, out_path: str, min_count: int):
    counter = Counter()

    #for file in os.listdir(in_path):
    #    filepath = os.path.join(in_path, file)
    filepath = in_path
    with open(filepath, "r", encoding="utf-8") as csvfile:
        reader = DictReader(csvfile, delimiter="\t")
        for row in reader:
            message = row["msg"].strip()
            # emote_ranges = str(row["emotes"]) + "/" + str(row["extemotes"])
            # emotenames = get_emotes_in_message(message, emote_ranges)
            emotenames = row["emotenames"].split()

            tuplelist = []
            for word in message.split():
                if word in emotenames:
                    continue
                tpl = tuple([word] + emotenames)
                tuplelist.append([tpl])
                # tuplelist.append([(word, emote)])

            # print(tuplelist)
            counter.update(chain(*tuplelist))

    c = {x[0]: {"emotes": x[1:], "count": count} for x, count in 
            sorted(counter.items(), key=lambda x: x[1], reverse=True) if count >= min_count}
    with open(out_path, "w") as outfile:
        json.dump(OrderedDict(c), outfile, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infiles_rootdir", type=str)
    parser.add_argument("-o", "--outdir_path")
    parser.add_argument("-m", "--min", type=int, default=100)
    args = parser.parse_args()

    indir = args.infiles_rootdir
    outdir = args.outdir_path

    build_vocabulary(indir, outdir, args.min)

    #build_vocabulary("../data/testdata/emote_filtered", "vocab.json", min_count=2)

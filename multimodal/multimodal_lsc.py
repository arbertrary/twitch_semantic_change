import os
import re
from csv import DictReader
from collections import Counter, OrderedDict
from itertools import chain
import argparse
import json


def get_emotes_in_message(message: str, emote_ranges):
    emotes = emote_ranges.split("/")

    emote_indices = [x.split(":")[1].strip() for x in emotes if x != ""]
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
        emotenames.append(emote.strip())

    return emotenames


# def get_emote_tuples(message: str, emotenames:[str], c:Counter):
#     words = message.split()
#
#     for emote in emotenames:
#         for word in words:
#             t = (word, emote)
#             c.update(t)

def build_vocabulary(in_path: str, out_path: str, min_count: int):
    counter = Counter()

    for file in os.listdir(in_path):
        filepath = os.path.join(in_path, file)
        with open(filepath, "r", encoding="utf-8") as csvfile:
            reader = DictReader(csvfile, delimiter=",")
            for row in reader:
                message = row["msg"].strip()
                emote_ranges = str(row["emotes"]) + "/" + str(row["extemotes"])
                emotenames = get_emotes_in_message(message, emote_ranges)

                tuplelist = []
                for emote in emotenames:
                    for word in message.split():
                        tuplelist.append([(word, emote)])

                # print(tuplelist)
                counter.update(chain(*tuplelist))

    print(counter.most_common(10))

    for c in counter.most_common():
        print(c)

    c = {str(x): count for x, count in counter.items() if count >= 100}
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

    # build_vocabulary("../data/testdata/emote_filtered", "vocab.json", min_count=100)

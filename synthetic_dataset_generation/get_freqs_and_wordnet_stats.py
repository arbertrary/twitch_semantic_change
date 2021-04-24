import sys
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "odenet"))

from odenet import *
from nltk.corpus import wordnet as wn
from collections import Counter
import argparse
import csv
import logging
import datetime


def read_chatlog_corpusfile(filepath: str, word_freq: Counter):
    with open(filepath, "r") as infile:
        for line in infile:
            msg = line.split()
            word_freq.update(msg)


def read_chatlog_csv(filepath: str, word_freq: Counter, emote_freq: Counter):
    with open(filepath, "r") as infile:
        reader = csv.DictReader(infile, delimiter=",")

        for row in reader:
            msg = row.get("msg").split()
            twemotes = row.get("emotes")
            extemotes = row.get("extemotes")

            emotes = twemotes.split("/") + extemotes.split("/")
            emotenames = [x.split(":")[0].strip() for x in emotes if len(x) != 0]
            if len(emotenames) != 0:
                emote_freq.update(emotenames)
            else:
                emote_freq.update(["NOEMOTE"])
            word_freq.update(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str,
                        default='/home/stud/bernstetter/datasets/synthetic_twitch/10p_sample_201911')
    parser.add_argument("-o", "--output_dir", type=str, default="/home/stud/bernstetter/datasets/synthetic_twitch")
    parser.add_argument("-l", "--lang", type=str, default="en")

    options = parser.parse_args()

    word_counter = Counter()
    emote_counter = Counter()

    out_dir = options.output_dir
    log_path = os.path.abspath(os.path.join(out_dir, "freqs_wn_stats.log"))
    logging.basicConfig(filename=log_path, encoding="utf-8", level=logging.DEBUG, format="%(message)s")
    logging.info("# Started at {}\n".format(datetime.datetime.now()))

    input_dir = os.path.abspath(options.input_dir)
    logging.info("# INPUT DIR: " + input_dir)
    for file in os.listdir(input_dir):
        with open(log_path, "r") as log:
            last = log.read().splitlines()
            if file in last:
                logging.info("Skipped " + file)
                continue

        file_path = os.path.join(input_dir, file)
        read_chatlog_corpusfile(file_path, word_counter)
        logging.info(file)
    #   read_chatlog_csv(file_path, word_counter, emote_counter)

    # emote_stats_path = os.path.join(out_dir, "emote_stats.csv")
    # with open(emote_stats_path, "w") as outfile:
    #    writer = csv.writer(outfile, delimiter=",")
    #    writer.writerow(["emote", "freq"])
    #    for item in emote_counter.most_common():
    #        (emote, freq) = item
    #        writer.writerow([emote, freq])

    vocab_stats_path = os.path.join(out_dir, "vocab_stats.csv")
    with open(vocab_stats_path, "w") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(["word", "freq", "n_senses", "senses", "n_hypernyms", "hypernyms", "n_hyponyms", "hyponyms"])
        for item in word_counter.most_common():
            (word, freq) = item
            if freq < 50:
                break

            if options.lang == "en":
                senses = wn.synsets(word)
                hypernyms = []
                hyponyms = []
                for sense in senses:
                    hypernyms.extend(sense.hypernyms())
                    hyponyms.extend(sense.hyponyms())
            else:
                try:
                    c = "BEFORE"
                    try:
                        (lemma_id, lemma_value, pos, lsenses) = check_word_lemma(word)
                    except TypeError:
                        senses = []
                        hypernyms = []
                        hyponyms = []
                        continue

                    lhypernyms = hypernyms_word(word)
                    lhyponyms = hyponyms_word(word)
                    c = "ERROR IS IN THE IF BLOCKS"

                    if len(lsenses) > 1:
                        senses = [s[1] for s in lsenses]
                    elif len(lsenses) == 0:
                        senses = []
                    else:
                        senses = [lsenses[0][1]]
                    c = "ERROR IS NOT IN FIRST IF BLOCK"

                    if len(lhypernyms) > 1:
                        hypernyms = [h[1] for h in lhypernyms]
                    elif len(lhypernyms) == 0:
                        hypernyms = []
                    else:
                        hypernyms = [lhypernyms[0][1]]
                    c = "ERROR IS NOT IN SND IF BLOCK"

                    if len(lhyponyms) > 1:
                        hyponyms = [h[1] for h in lhyponyms]
                    elif len(lhyponyms) == 0:
                        hyponyms = []
                    else:
                        hyponyms = [lhyponyms[0][1]]
                except TypeError as e:
                    print(e)
                    print(c)
                    print(word)
                    print((lemma_id, lemma_value, pos, lsenses))
                    print(lhypernyms)
                    print(lhyponyms)
                    exit()



            n_senses = len(senses)
            n_hypernyms = len(set(hypernyms))
            n_hyponyms = len(set(hyponyms))

            senses = ' '.join(str(x) for x in senses)
            hypernyms = ' '.join(str(x) for x in set(hypernyms))
            hyponyms = ' '.join(str(x) for x in set(hyponyms))
            writer.writerow([word, str(freq), str(n_senses), senses, str(n_hypernyms), hypernyms, str(n_hyponyms),
                             hyponyms])

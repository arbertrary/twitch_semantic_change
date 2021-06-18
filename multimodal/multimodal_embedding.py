import argparse
import csv
import os
import gensim

assert gensim.models.word2vec.FAST_VERSION > -1
from time import time
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ChatYielder(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                yield row[0].split()


def generate(vector_size, window_size, min_count, no_of_iter, workers, corpus_location, model_save_location, skipgram):
    params = {'sg': skipgram, 'size': vector_size, 'window': window_size, 'min_count': min_count, 'iter': no_of_iter,
              'workers': workers, 'sample': 1E-5}

    if sg == 1:
        params["negative"] = 5
        params["hs"] = 0

    model = gensim.models.Word2Vec(**params)

    t = time()

    chat_messages = ChatYielder(corpus_location)
    model.build_vocab(sentences=chat_messages, progress_per=50000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()

    model.train(sentences=chat_messages, total_examples=model.corpus_count, queue_factor=100, epochs=model.iter,
                report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    output_dir = os.path.join(model_save_location,
                              "vec_" + str(params['size']) + "_w" + str(params['window']) + "_mc" + str(
                                  params['min_count']) + "_iter" + str(params['iter']) + "_sg" + str(
                                  params['sg']))

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "saved_model.gensim")

    print("# Model saved to: ", output_filepath)
    model.save(output_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generating an embedding")

    # Input and output directories

    parser.add_argument("-in", "--input_dir", default="/home/stud/bernstetter/datasets/twitch/",
                        help="The input directory. Is required to be either a directory with text files or a single corpus file")
    parser.add_argument("-m", "--model_dir", default="/home/stud/bernstetter/models/twitch/sorted/")
    # Defaults taken from Jan Pfister's code for Emote-Controlled
    # Gensim model Settings
    parser.add_argument("-wc", "--word_min_count", type=int, default=100,
                        help="min number of occurences for vocabulary")
    parser.add_argument("-wrk", "--worker_count", type=int, default=6,
                        help="number of workers for parallelization")
    parser.add_argument("-e", "--vector_size", type=int, default=128,
                        help="number of embedding dimensions")
    parser.add_argument("-win", "--window_size", type=int, default=5, help="The Window size")
    parser.add_argument("-ep", "--epoch_count", default=10, help="no of iteration")
    parser.add_argument("-sg", "--skipgram", type=int, default=0, help="1 = skipgram, 0 = CBOW")

    # Settings for this script

    args = vars(parser.parse_args())
    print(args)

    vs = int(args["vector_size"])
    ws = int(args["window_size"])
    n_mc = int(args["word_min_count"])
    n_ep = int(args["epoch_count"])
    n_workers = int(args["worker_count"])
    sg = int(args["skipgram"])

    input_dir = args["input_dir"]

    model_dir = args["model_dir"]

    print(input_dir)
    print(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    generate(vs, ws, n_mc, n_ep, n_workers, input_dir, model_dir, sg)

import argparse
from collections import Counter
import os
import string
import gensim
from gensim.test.utils import datapath

assert gensim.models.word2vec.FAST_VERSION > -1
from time import time
from loader import ChatYielder
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Parameter von Jan beim twitch paper
# vector size: w2v_dims default=128
# window = 5
# min_count: default = 100
# iter = 1
# number of workers: default 8

def clean(line: str) -> [str]:
    msg = [word for word in line.split() if
           not (word.startswith("!") or word.startswith("@") or word.startswith("http"))]
    msg = list(map(lambda x: x[:-1] if x[-1] in string.punctuation else x, msg))

    return msg


def generate(vector_size, window_size, min_count, no_of_iter, workers, corpus_location, model_save_location, skipgram, word2vec, lowercase, cleaned):
    params = {'sg': skipgram, 'size': vector_size, 'window': window_size, 'min_count': min_count, 'iter': no_of_iter,
              'workers': workers, 'sample': 1E-5}

    if sg == 1:
        params["negative"]=5
        params["hs"] = 0
        

    #print("# Running independently for timestep {}".format(timestep))
    # csvfiles = [os.path.join(single_input_dir, p) for p in os.listdir(single_input_dir)]

    if w2v == 1:
        # model = gensim.models.word2vec.Word2Vec(size=vs, window=ws, min_count=n_mc, iter=n_ep, workers=n_workers)
        model = gensim.models.Word2Vec(**params)
    else:
        # model = gensim.models.fasttext.FastText(size=vs, window=ws, min_count=n_mc, iter=n_ep, workers=n_workers)
        model = gensim.models.fasttext.FastText(**params)

    t = time()

    if os.path.isdir(corpus_location):
        #chat_messages = ChatYielder(corpus_location)
        chat_messages = gensim.models.word2vec.PathLineSentences(corpus_location)
        model.build_vocab(sentences=chat_messages, progress_per=10000)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()

        model.train(sentences=chat_messages, total_examples=model.corpus_count, queue_factor=100, epochs=model.iter,
                    report_delay=1)
    else:
        model.build_vocab(corpus_file=corpus_location, progress_per=10000)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()

        model.train(corpus_file=corpus_location, total_words=model.corpus_total_words,
                    total_examples=model.corpus_count, queue_factor=100, epochs=model.iter,
                    report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    
    output_dir = os.path.join(model_save_location,
                              "vec_" + str(params['size']) + "_w" + str(params['window']) + "_mc" + str(
                                  params['min_count']) + "_iter" + str(params['iter']) + "_sg" + str(
                                  params['sg']) + "_lc" + str(lowercase) + "_clean" + str(cleaned) + "_w2v" + str(
                                  word2vec))

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "saved_model.gensim")

    print("# Model saved to: ", output_filepath)
    model.save(output_filepath)


if __name__ == '__main__':
    # python3 embedding_generation.py -i "../testdata/synthetic/subsampled_70/" -m "../testdata/synthetic/models/" -s 1 -wc 1

    parser = argparse.ArgumentParser(description="generating an embedding")

    # Input and output directories

    parser.add_argument("-in", "--input_dir", default="/home/stud/bernstetter/datasets/twitch/",
                        help="The input directory. Is required to be either a directory with text files or a single corpus file")
    parser.add_argument("-m", "--model_dir", default="/home/stud/bernstetter/models/twitch/sorted/")
    # Defaults taken from Jan Pfister's code for Emote-Controlled
    # Gensim model Settings
    parser.add_argument("-w2v", "--word2vec", type=int, default=1, help="Prefer fasttext")
    parser.add_argument("-wc", "--word_min_count", type=int, default=100,
                        help="min number of occurences for vocabulary")
    parser.add_argument("-wrk", "--worker_count", type=int, default=6,
                        help="number of workers for parallelization")
    parser.add_argument("-e", "--vector_size", type=int, default=128,
                        help="number of embedding dimensions")
    parser.add_argument("-win", "--window_size", type=int, default=5, help="The Window size")
    parser.add_argument("-ep", "--epoch_count", default=1, help="no of iteration")
    parser.add_argument("-sg", "--skipgram", type=int, default=0, help="1 = skipgram, 0 = CBOW")

    # Settings for this script

    parser.add_argument("-lc", "--lower_case", type=int, default=0,
                        help="0=don't lowercase messages; 1=lowercase all messages")
    parser.add_argument("-cl", "--clean", type=int, default=0,
                        help="0=keep original message, 1=clean messages from e.g. !chatcommand, urls, name mentions")
    args = vars(parser.parse_args())
    print(args)

    args_known, leftovers = parser.parse_known_args()

    lc = args["lower_case"]
    cl = args["clean"]
    # if args["lower_case"] == 0:
    #     lower = False
    # else:
    #     lower = True

    w2v = args["word2vec"]

    vs = int(args["vector_size"])
    ws = int(args["window_size"])
    n_mc = int(args["word_min_count"])
    n_ep = int(args["epoch_count"])
    n_workers = int(args["worker_count"])
    sg = args["skipgram"]

    input_dir = args["input_dir"]

    model_dir = args["model_dir"]
    print(input_dir)
    print(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # If only one month is given train embeddings for single month
    # This is done in this way to let multiple Kubernetes Jobs train embeddings for one month each
    # else train continuous embeddings using previous months

    generate(vs, ws, n_mc, n_ep, n_workers, input_dir, model_dir, sg, w2v, lc, cl)

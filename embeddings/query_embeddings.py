import gensim
import os
from gensim.test.utils import datapath
import argparse
import gensim.downloader as api
from pprint import pprint

def get_related_terms(model, token, topn=10):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """
    try:
        for word, similarity in model.most_similar(positive=[token], topn=topn):
            print(word, round(similarity, 3))
    except:
        print("Error!")


def most_similar(model):
    while True:
        word = input("Enter a word (QUIT to exit):")
        if word == "QUIT":
            break
        print("====================")
        print(get_related_terms(model, word))
        print("====================")


def retrieve_model(model_path):
    m = os.path.join(model_path)

    if "w2v1" in m:
        model = gensim.models.Word2Vec.load(m)
    else:
        model = gensim.models.FastText.load(m)
    model = model.wv
    model.init_sims(replace=True)

    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-mp", "--model_path", help="path to saved word embedding model")
    ap.add_argument("-ms", "--most_similar", required=False, default="1", help="0 for no most similar list")

    args = vars(ap.parse_args())
    args_known, leftovers = ap.parse_known_args()

    mp = args["model_path"]
    word_vectors = retrieve_model(mp)
    print(word_vectors["Pog"])
    print(word_vectors.doesnt_match("LUL WutFace DansGame".split()))


    if int(args_known.most_similar) == 1:
        most_similar(word_vectors)

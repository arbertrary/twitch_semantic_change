import torch
import argparse
from scipy.spatial.distance import cosine
import os
import numpy as np



def align_models(model1, model2, config):
    """
    Align model2 to model1
    """
    intersection_vocab = list(set.intersection(set([x for x in model1]), set([x for x in model2])))
    # generate matrices for both
    # with the words/indices being the same
    alignment_reference_matrix = np.empty((len(intersection_vocab), config["latent_dims"]))
    other_matrix = np.empty((len(intersection_vocab), ["latent_dims"]))

    for i, word in enumerate(intersection_vocab):
        alignment_reference_matrix[i] = (model1[word].detach().cpu().numpy())
        other_matrix[i] = (model2[word].detach().cpu().numpy())

    m = other_matrix.T.dot(alignment_reference_matrix)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)
    other_matrix = other_matrix.dot(ortho)

    return intersection_vocab, alignment_reference_matrix, other_matrix


def rank_by_cosine(model1, model2, n, config):
    dists = []

    intersection_set, base_matrix, other_matrix = align_models(model1, model2, config)

    for i, word in enumerate(intersection_set):
        dist = cosine(base_matrix[i], other_matrix[i])
        dists.append((word, dist))

    dists.sort(key=lambda x: x[1], reverse=True)

    return dists[:n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_filepath", type=str,
                        help="path to first embedding model")
    parser.add_argument("--model2_filepath", type=str,
                        help="path to second embedding model")
    parser.add_argument("-t", "--t_best", type=int, default=20, help="Number of top-ranked words to output")
    parser.add_argument("-o", "--outfiles_dir", type=str,
                        help="Path to file where results will be written")
    parser.add_argument("--latent_dim", type=int, default=128)

    options = parser.parse_args()
    CONFIG = {"latent_dims": options.latent_dim}

    model1 = torch.load(options.model1_filepath)
    model2 = torch.load(options.model2_filepath)
    # model1 = torch.load("../data/testdata/testmodels/fused_embeddings.pt")
    # model2 = torch.load("../data/testdata/testmodels/fused_embeddings.pt")
    n_best_by_cosine = rank_by_cosine(model1, model2, n=options.t_best, config=CONFIG)

    print("Done ranking by cosine distance measure.")

    os.makedirs(options.outfiles_dir, exist_ok=True)
    outfilepath = options.outfiles_dir + '/cosine.tsv'
    # outfilepath = "cosine.tsv"

    with open(outfilepath, 'w', encoding="utf-8") as outfile:
        for (i, item) in enumerate(n_best_by_cosine):
            outfile.write('{}\t{}\t{}\n'.format(i, item[0], item[1]))

    # write_logfile(outfilepath, options, start_time)
    # print(" Output and log file written to {}".format(options.outfiles_dir)

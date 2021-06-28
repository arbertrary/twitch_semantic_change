#!/usr/bin/python3

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile, datapath
import numpy as np
import argparse
import string
from scipy.spatial.distance import cosine
import datetime
import sys
import os
import datetime
import csv


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's vectors and vectors_norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).

    [This implementation by Ryan Heuser (quadrismegistus): https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf]
    """

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # get the embedding matrices
    base_vecs = in_base_embed.vectors_norm
    other_vecs = in_other_embed.vectors_norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs)
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (vectors_norm)by "ortho"
    other_embed.vectors_norm = other_embed.vectors = (other_embed.vectors_norm).dot(ortho)
    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new vectors and vectors_norm objects in both gensim models:
        -- so that Row 0 of m1.vectors will be for the same word as Row 0 of m2.vectors
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    [This implementation by Ryan Heuser (quadrismegistus): https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf]
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.vocab.keys())
    vocab_m2 = set(m2.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        # sys.stdout.write('VOCAB IDENTICAL')
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.vocab[w].count + m2.vocab[w].count, reverse=True)

    # Then for each model...
    for m in [m1, m2]:
        # Replace old vectors_norm array with new one (with common vocab)
        indices = [m.vocab[w].index for w in common_vocab]
        old_arr = m.vectors_norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.vectors_norm = m.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.keyedvectors.Vocab(index=new_index, count=old_vocab_obj.count)
        m.vocab = new_vocab

    return (m1, m2)


def measure_semantic_shift_by_neighborhood(model1, model2, word, k, verbose=False):
    """
    Basic implementation of William Hamilton (@williamleif) et al's measure of semantic change
    proposed in their paper "Cultural Shift or Linguistic Drift?" (https://arxiv.org/abs/1606.02821),
    which they call the "local neighborhood measure." They find this measure better suited to understand
    the semantic change of nouns owing to "cultural shift," or changes in meaning "local" to that word,
    rather than global changes in language ("linguistic drift") use that are better suited to a
    Procrustes-alignment method (also described in the same paper.)

    Arguments are:
    - `model1`, `model2`: Are gensim word2vec models.
    - `word` is a sting representation of a given word.
    - `k` is the size of the word's neighborhood (# of its closest words in its vector space).
    [This implementation by Ryan Heuser (quadrismegistus): https://gist.github.com/quadrismegistus/15cafbdd878a98b060ef910c843fcf5a]
    """

    # Check that this word is present in both models
    if not word in model1.vocab or not word in model2.vocab:
        sys.stdout.write("!! Word %s not present in both models." % word)
        return None

    # Get the two neighborhoods
    neighborhood1 = [w for w, c in model1.most_similar(word, topn=k)]
    neighborhood2 = [w for w, c in model2.most_similar(word, topn=k)]

    # sys.stdout.write?
    if verbose:
        sys.stdout.write('>> Neighborhood of associations of the word "%s" in model1:' % word)
        sys.stdout.write(', '.join(neighborhood1))
        sys.stdout.write()
        sys.stdout.write('>> Neighborhood of associations of the word "%s" in model2:' % word)
        sys.stdout.write(', '.join(neighborhood2))

    # Get the 'meta' neighborhood (both combined)
    meta_neighborhood = list(set(neighborhood1) | set(neighborhood2))

    # Filter the meta neighborhood so that it contains only words present in both models
    meta_neighborhood = [w for w in meta_neighborhood if w in model1.vocab and w in model2.vocab]

    # For both models, get a similarity vector between the focus word and all of the words in the meta neighborhood
    vector1 = [model1.similarity(word, w) for w in meta_neighborhood]
    vector2 = [model2.similarity(word, w) for w in meta_neighborhood]

    # Compute the cosine distance *between* those similarity vectors
    dist = cosine(vector1, vector2)

    # Return this cosine distance -- a measure of the relative semantic shift for this word between these two models
    return dist


def rank_by_cosine(model1, model2, model1_filepath, model2_filepath, n, f, target_words):
    (model1, model2) = intersection_align_gensim(model1, model2)
    # sys.stdout.write('\nLIMITED TO COMMON VOCABULARY')

    model2 = smart_procrustes_align_gensim(model1, model2)
    # sys.stdout.write('\nALIGNED EMBEDDINGS')

    dists = []
    culled_vocab = [(word, model1.vocab[word].count) for word in model1.vocab]
    culled_vocab.sort(key=lambda tup: tup[1])
    culled_vocab = culled_vocab[int(len(culled_vocab) * f):]

    culled_vocab2 = [(word, model2.vocab[word].count) for word in model2.vocab]
    culled_vocab2.sort(key=lambda tup: tup[1])
    culled_vocab2 = culled_vocab2[int(len(culled_vocab2) * f):]

    intersection_set = set.intersection(set([x[0] for x in culled_vocab]), set([x[0] for x in culled_vocab2]))

    for word in intersection_set:
        if len(target_words) != 0 and word not in target_words:
            continue

        if (word.startswith("!") or word.startswith("@") or word.startswith("http") or (
                word[-1] in string.punctuation) or (word[0] in string.punctuation)):
            continue
        # if model1.vocab[word].count > f and model2.vocab[word].count > f:
        dist = cosine(model1[word], model2[word])
        dists.append((word, dist))

    # sys.stdout.write('\nGOT DISTS')

    dists.sort(key=lambda x: x[1], reverse=True)
    # sys.stdout.write('\nSORTED DISTS')

    sys.stdout.write("\n{}\n Model1: {}\n Model2: {}".format(datetime.datetime.now(), model1_filepath, model2_filepath))
    sys.stdout.write("\n Params:-  f: {}  t: {}\n\n".format(f, n))
    sys.stdout.write("\n Length of 10p culled vocab = {}\n\n".format(len(culled_vocab)))
    thresh100 = [word for word in model1.vocab if model1.vocab[word].count > 100 and model2.vocab[word].count > 100]
    thresh500 = [word for word in model1.vocab if model1.vocab[word].count > 500 and model2.vocab[word].count > 500]
    thresh1000 = [word for word in model1.vocab if model1.vocab[word].count > 1000 and model2.vocab[word].count > 1000]
    sys.stdout.write("\n Length of threshold 100 vocab = {}\n\n".format(len(thresh100)))
    sys.stdout.write("\n Length of threshold 500 vocab = {}\n\n".format(len(thresh500)))
    sys.stdout.write("\n Length of threshold 1000 vocab = {}\n\n".format(len(thresh1000)))
    sys.stdout.write("\n Length of intersection set = {}\n\n".format(len(intersection_set)))

    sys.stdout.write("\n\n{}".format(culled_vocab))

    # sys.stdout.write('\n============================')
    # sys.stdout.write('\n============================')
    # sys.stdout.write('\nTOP WORDS BY COSINE DIST')
    # sys.stdout.write('\n============================')
    # sys.stdout.write('\n============================')
    # sys.stdout.write('\n\n')

    # for d in dists[:n]:
    #    sys.stdout.write(d[0])
    #    sys.stdout.write('\n============================')
    #    sys.stdout.write('\nModel1: ' + ' '.join([w for w, c in model1.most_similar(d[0], topn=10)]))
    #    sys.stdout.write('\nModel2: ' + ' '.join([w for w, c in model2.most_similar(d[0], topn=10)]))
    #    sys.stdout.write('\n\n')

    return dists[:n]


def rank_by_neighbourhood_shift(model1, model2, model1_filepath, model2_filepath, n, k, f, target_words):
    (model1, model2) = intersection_align_gensim(model1, model2)
    # sys.stdout.write('\nLIMITED TO COMMON VOCABULARY')

    dists = []
    culled_vocab = [(word, model1.vocab[word].count) for word in model1.vocab]
    culled_vocab.sort(key=lambda tup: tup[1])
    culled_vocab = culled_vocab[int(len(culled_vocab) * f):]

    culled_vocab2 = [(word, model2.vocab[word].count) for word in model2.vocab]
    culled_vocab2.sort(key=lambda tup: tup[1])
    culled_vocab2 = culled_vocab2[int(len(culled_vocab2) * f):]

    intersection_set = set.intersection(set([x[0] for x in culled_vocab]), set([x[0] for x in culled_vocab2]))

    for word in intersection_set:
        if len(target_words) != 0 and word not in target_words:
            continue

        if (word.startswith("!") or word.startswith("@") or word.startswith("http") or (
                word[-1] in string.punctuation) or (word[0] in string.punctuation)):
            continue
        # if model1.vocab[word].count > f and model2.vocab[word].count > f:
        dist = measure_semantic_shift_by_neighborhood(model1, model2, word, k, verbose=False)
        dists.append((word, dist))
    # sys.stdout.write('\nGOT DISTS')

    dists.sort(key=lambda x: x[1], reverse=True)
    # sys.stdout.write('\nSORTED DISTS')

    sys.stdout.write("\n{}\n Model1: {}\n Model2: {}".format(datetime.datetime.now(), model1_filepath, model2_filepath))
    sys.stdout.write("\n Params:-  f: {}  t: {}  k: {}\n\n".format(f, n, k))

    sys.stdout.write('\n============================')
    sys.stdout.write('\n============================')
    sys.stdout.write('\nTOP WORDS BY NEIGHBOURHOOD SHIFT')
    sys.stdout.write('\n============================')
    sys.stdout.write('\n============================')
    sys.stdout.write('\n\n')

    for d in dists[:n]:
        sys.stdout.write(d[0])
        sys.stdout.write('\n============================')
        sys.stdout.write('\nModel1: ' + ' '.join([w for w, c in model1.most_similar(d[0], topn=k)]))
        sys.stdout.write('\nModel2: ' + ' '.join([w for w, c in model2.most_similar(d[0], topn=k)]))
        sys.stdout.write('\n\n')

    return dists[:n]


def write_logfile(outfilepath, options, start_time):
    logfile_path = outfilepath + '.log'
    with open(logfile_path, 'w') as logfile:
        logfile.write('Script started at: {}\n\n'.format(start_time))
        logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
        logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
        logfile.write('Options used:-\n')
        for (option, value) in vars(options).items():
            logfile.write('{}\t{}\n'.format(option, value))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_filepath", type=str,
                        default="/models/continuous/3/2012-01_2012-03/vec_200_w10_mc500_iter15_sg0/saved_model.gensim",
                        help="path to first embedding model")
    parser.add_argument("--model2_filepath", type=str,
                        default="/models/continuous/3/2017-04_2017-06/vec_200_w10_mc500_iter15_sg0/saved_model.gensim",
                        help="path to second embedding model")
    parser.add_argument("-f", "--frequency_threshold", type=float, default=0.1,
                        help="Limit the words we rank to only those which occured at least f times in both months")
    parser.add_argument("-k", "--k_neighbors", type=int, default=25,
                        help="Number of neighbors to consider if using neighborhood shift measure")
    parser.add_argument("-t", "--t_best", type=int, default=20, help="Number of top-ranked words to output")
    parser.add_argument("-c", "--cosine", action="store_true", default=False,
                        help="Include this flag to rank by cosine dist measure (include both this and the "
                             "neighborhood flag to produce separate rankings using each measure)")
    parser.add_argument("-n", "--neighborhood", action="store_true", default=False,
                        help="Include this flag to rank by neighborhood shift measure (include both this and the "
                             "cosine flag to produce separate rankings using each measure)")
    parser.add_argument("-o", "--outfiles_dir", type=str,
                        default="/results/two_step_candidates/continuous/3/2012-01_2012-03_to_2017-04_2017-06/vec_200_w10_mc500_iter15_sg0/",
                        help="Path to file where results will be written")
    parser.add_argument("--targets", type=str,
                        help="Path to a csv (or single column) file with target words. If present, only those will be cosidered")
    parser.add_argument("-g", "--glove", action="store_true", default=False)
    options = parser.parse_args()

    start_time = datetime.datetime.now()

    targets_temp = []
    target_path = options.targets
    if target_path and os.path.isfile(target_path):
        with open(target_path, "r") as targetsfile:
            reader = csv.reader(targetsfile, delimiter=",")
            for row in reader:
                targets_temp.append(row[0])
    elif target_path and os.path.isdir(target_path):
        for file in os.listdir(target_path):
            with open(os.path.join(target_path, file), "r") as targetsfile:
                reader = csv.reader(targetsfile, delimiter=",")
                for row in reader:
                    targets_temp.append(row[0])
    else:
        pass

    targets = set(targets_temp)
    print(targets)

    if not options.cosine and not options.neighborhood and not options.cosine_unaligned:
        raise RuntimeError(
            'Please specify which semantic change measure(s) to use by including at least one of --cosine, '
            '--cosine_unaligned, or --neighborhood.')
    else:

        if options.glove:
            tmp_file1 = get_tmpfile("w2v_model1.txt")
            glove_file1 = datapath(options.model1_filepath)
            tmp_file2 = get_tmpfile("w2v_model2.txt")
            glove_file2 = datapath(options.model2_filepath)

            _ = glove2word2vec(glove_file1, tmp_file1)
            model1 = gensim.models.KeyedVectors.load_word2vec_format(tmp_file1)

            _ = glove2word2vec(glove_file2, tmp_file2)
            model2 = gensim.models.KeyedVectors.load_word2vec_format(tmp_file2)
        else:
            model1 = gensim.models.Word2Vec.load(options.model1_filepath)
            model2 = gensim.models.Word2Vec.load(options.model2_filepath)

        model1 = model1.wv
        model1.init_sims(replace=True)
        model2 = model2.wv
        model2.init_sims(replace=True)

        if options.cosine:
            n_best_by_cosine = rank_by_cosine(model1, model2, options.model1_filepath, options.model2_filepath,
                                              n=options.t_best, f=options.frequency_threshold, target_words=targets)

            print("Done ranking by cosine distance measure.")

            os.makedirs(options.outfiles_dir, exist_ok=True)
            outfilepath = options.outfiles_dir + '/cosine.tsv'

            with open(outfilepath, 'w') as outfile:
                for (i, item) in enumerate(n_best_by_cosine):
                    outfile.write('{}\t{}\t{}\n'.format(i, item[0], item[1]))

            write_logfile(outfilepath, options, start_time)
            print(" Output and log file written to {}".format(options.outfiles_dir))

        if options.neighborhood:
            n_best_by_neighborhood = rank_by_neighbourhood_shift(model1, model2, options.model1_filepath,
                                                                 options.model2_filepath, n=options.t_best,
                                                                 k=options.k_neighbors, f=options.frequency_threshold,
                                                                 target_words=targets)

            print("Done ranking by neighborhood shift measure.")

            os.makedirs(options.outfiles_dir, exist_ok=True)
            outfilepath = options.outfiles_dir + '/neighborhood.tsv'

            with open(outfilepath, 'w') as outfile:
                for (i, item) in enumerate(n_best_by_neighborhood):
                    outfile.write('{}\t{}\t{}\n'.format(i, item[0], item[1]))

            write_logfile(outfilepath, options, start_time)
            print("Output and log file written to {}".format(options.outfiles_dir))

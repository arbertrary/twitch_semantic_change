#!/usr/bin/python3

import gensim
import argparse
import os
import hamilton_semantic_change_measures as hs
from scipy.spatial.distance import cosine
import numpy as np
import datetime
from collections import Counter, defaultdict
import json
import string
import glob
import itertools
import csv
import multiprocessing
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile, datapath
import random


def load_model(model_path):
    """
    Load the trained gensim word embedding model stored at model_path. Since we don’t need the full model state any
    more (don’t need to continue training), the state can be discarded, and we just return the trained vectors (i.e.
    the KeyedVectors instance, model.wv). We call init_sims() to precompute L2-normalized vectors,
    using 'replace=True' to forget the original vectors and only keep the normalized ones (saves lots of memory).
    """
    if options.glove:
        print("LOADING GLOVE MODEL")
        n = random.randint(1,2000)
        tmp_file = get_tmpfile(str(n)+".txt")
        glove_file = datapath(model_path)

        _ = glove2word2vec(glove_file, tmp_file)
        m = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
    else:
        m = gensim.models.Word2Vec.load(model_path)
    
    m = m.wv
    m.init_sims(replace=True)
    
    return m


def get_dist_dict(model_path, alignment_reference_model_path, comparison_reference_model_path, vocab, distance_measure,
                  k, training_mode):
    """
    Return a dictionary which contains, for each word which appears in the intersection of the vocabularies
    of the current timestep's model,
    the alignment reference model (if applicable),
    and the comparison reference model,
    the distance between that word's representation in the current timestep's model
    and its representation in the comparison reference model.
    """

    # Load the model for the current timestep, the model we are aligning everything to, and the model we are
    # comparing everything to.
    model = load_model(model_path)
    alignment_reference_model = load_model(alignment_reference_model_path)
    comparison_reference_model = load_model(comparison_reference_model_path)

    # Align both the current timestep's model and the comparison reference model to the alignment reference model.
    if distance_measure == 'cosine' and training_mode == 'independent':

        if alignment_reference_model_path != comparison_reference_model_path:
            # print('ALIGNING {} to {}.'.format(comparison_reference_model_path, alignment_reference_model_path))
            comparison_reference_model = hs.smart_procrustes_align_gensim(alignment_reference_model,
                                                                          comparison_reference_model)
            alignment_reference_model = load_model(alignment_reference_model_path)

        # print('ALIGNING {} to {}.\n\n'.format(model_path, alignment_reference_model_path))
        model = hs.smart_procrustes_align_gensim(alignment_reference_model, model)

    # This will be a dictionary with keys = words, values = distance between the word's vector in the current
    # timestep and its vector in the comparison reference model.
    dist_dict = {}

    for word in vocab:
        if (word.startswith("!") or word.startswith("@") or word.startswith("http") or (
                word[-1] in string.punctuation) or (word[0] in string.punctuation)):
            continue

        if word in comparison_reference_model and word in model:
            if distance_measure == 'cosine':
                # print('cosine')
                dist_dict[word] = cosine(comparison_reference_model[word], model[word])
            else:  # distance_measure == 'neighborhood':
                dist_dict[word] = hs.measure_semantic_shift_by_neighborhood(comparison_reference_model, model, word, k)

        # if the word does not occur in both the current timestep's model and the comparison reference model's vocab
        # (and implicitly, in the alignment reference model, since we aligned them both to that),
        # then we can't calculate a distance measure for this word and this timestep.
        else:
            pass
    # dist_dict[word] = None
    # else:
    # 	raise RunTimeError("Invalid command line argument: Only possible values for option -m (--distance_measure) are 'cosine' or 'neighborhood'")

    return dist_dict


def get_dist_dict_multithreaded(tup):
    (i, model_path) = tup
    if i == 0 and (options.compare_to == 'previous' or options.align_to == 'previous' or options.compare_to == 'first'):
        return None

    elif i == len(model_paths) - 1 and options.compare_to == 'last':
        return None

    else:

        if options.align_to == 'first':
            alignment_reference_model_path = model_paths[0]
        elif options.align_to == 'last':
            alignment_reference_model_path = model_paths[-1]
        else:
            alignment_reference_model_path = model_paths[i - 1]

        if options.compare_to == 'first':
            comparison_reference_model_path = model_paths[0]
        elif options.compare_to == 'last':
            comparison_reference_model_path = model_paths[-1]
        else:
            comparison_reference_model_path = model_paths[i - 1]

        dist_dict = get_dist_dict(model_path, alignment_reference_model_path, comparison_reference_model_path, vocab,
                                  options.distance_measure, options.k_neighbors, options.training_mode)

        return (i, dist_dict)


def get_z_score_dict(dist_dict):
    """
    Convert the dictionary of distance scores for a given timestep into a dictionary of z-scores
    - i.e. how many standard deviations is a given word's distance score from the mean of all word's distance scores at this timestep?
    """

    # calculate mean and variance of distance scores, ignoring any words for whom the value was None
    # -- calculate the mean and variance of the distance scores for all words
    # which are represented in both the current timestep's model and the comparison reference model.
    mean = np.mean(list(dist_dict.values()))
    var = np.var(list(dist_dict.values()))

    z_score_dict = {}
    for word in dist_dict:
        z_score_dict[word] = (dist_dict[word] - mean) / np.sqrt(var)

    return z_score_dict


def compute_mean_shift(time_series_dict, j, compare_to):
    """
    Compute the mean_shift score at index j of the given time-series.
    """

    timestep_to_index = {}
    for (i, timestep) in enumerate(sorted(time_series_dict.keys())):
        timestep_to_index[timestep] = i

    xs = list(itertools.chain.from_iterable(
        [timestep_to_index[timestep]] * len(time_series_dict[timestep]) for timestep in time_series_dict))
    ys = list(itertools.chain.from_iterable(time_series_dict.values()))

    if compare_to == 'first':
        return np.mean([ys[i] for i in range(len(ys)) if xs[i] > j]) - np.mean(
            [ys[i] for i in range(len(ys)) if xs[i] <= j])
    else:  # compare_to == 'last' or compare_to == 'previous':
        return np.mean([ys[i] for i in range(len(ys)) if xs[i] <= j]) - np.mean(
            [ys[i] for i in range(len(ys)) if xs[i] > j])


def get_mean_shift_series(time_series_dict, compare_to):
    """
    Compute a given word's mean_shift time-series from its time-series of z-scores.
    """
    return [compute_mean_shift(time_series_dict, j, compare_to) for j in range(len(time_series_dict.keys()) - 1)]


def get_p_value_series(word, mean_shift_series, n_samples, z_scores_dict, compare_to):
    """
    Randomly permute the z-score time series n_samples times, and for each
    permutation, compute the mean-shift time-series of those permuted z-scores,
    and at each index, check if the mean-shift score from the permuted series is greater than the mean-shift score
    from the original series.
    The p-value is the proportion of randomly permuted series which yielded
    a mean-shift score greater than the original mean-shift score.
    """
    p_value_series = np.zeros(len(mean_shift_series))
    for i in range(n_samples):
        permuted_z_score_series = np.random.permutation(list(z_scores_dict.values()))
        permuted_z_scores_dict = {}
        for (i, z_scores) in enumerate(permuted_z_score_series):
            permuted_z_scores_dict[i] = z_scores
        mean_shift_permuted_series = get_mean_shift_series(permuted_z_scores_dict, compare_to)

        for x in range(len(mean_shift_permuted_series)):
            if mean_shift_permuted_series[x] > mean_shift_series[x]:
                p_value_series[x] += 1
    p_value_series /= n_samples
    return p_value_series


def detect_change_point(word, z_scores_dict, n_samples, p_value_threshold, gamma_threshold, compare_to):
    """
    This function computes the mean-shift time-series from the given word's z-score series, then computes the p-value series,
    """

    index_to_timestep = {}
    for (i, timestep) in enumerate(sorted(z_scores_dict.keys())):
        index_to_timestep[i] = timestep

    mean_shift_series = get_mean_shift_series(z_scores_dict, compare_to)

    p_value_series = get_p_value_series(word, mean_shift_series, n_samples, z_scores_dict, compare_to)

    # set p-values for any time-slices with average z-scores below gamma threshold to 1, so that these time-slices won't get chosen.
    for i in range(len(p_value_series)):
        if np.mean(z_scores_dict[index_to_timestep[i]]) < gamma_threshold:
            p_value_series[i] = 1

    # find minimum p_value:
    p_value_series = np.array(p_value_series)
    try:
        min_p_val = p_value_series.min()
    except ValueError:
        print(word)
        print(z_scores_dict)
        print(mean_shift_series)
        print(p_value_series)

    # if minimum p_value is below the threshold:
    if min_p_val < p_value_threshold:

        # get indices of time-slices with minimum p_value:
        indices = np.where(p_value_series == min_p_val)[0]

        # as a tie-breaker, return the one which corresponds to the biggest mean_shift
        (change_point, mean_shift) = max([(i, mean_shift_series[i]) for i in indices], key=lambda x: x[1])

        z_score = np.mean(z_scores_dict[index_to_timestep[change_point]])
        time_slice_label = index_to_timestep[change_point]

        return (word, time_slice_label, min_p_val, mean_shift, z_score)

    else:
        return None


def get_word_change_point(tup):
    (word, dists_dict) = tup

    change_point = detect_change_point(word, dists_dict, options.n_samples, options.p_value_threshold,
                                       options.gamma_threshold, options.compare_to)

    if change_point:
        return change_point


def write_logfile(outfilepath, options, start_time):
    logfile_path = outfilepath + '.log'
    with open(logfile_path, 'w') as logfile:
        logfile.write('Script started at: {}\n\n'.format(start_time))
        logfile.write('Output created at: {}\n\n'.format(datetime.datetime.now()))
        logfile.write('Script used: {}\n\n'.format(os.path.abspath(__file__)))
        logfile.write('Options used:- {}\n')
        for (option, value) in vars(options).items():
            logfile.write('{}\t{}\n'.format(option, value))


if __name__ == "__main__":
    # OUTPUT FIELDS:  (word, time_slice_label, min_p_val, mean_shift, z_score)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--models_rootdir", type=str, default="/home/stud/bernstetter/models/synthetic_twitch/",
                        help="path to directory where models are stored")
    # parser.add_argument("-f", "--first_timeslice", type=str, default='2019-05', help="which timeslice to start from")
    # parser.add_argument("-l", "--last_timeslice", type=str, default='2020-04', help="which timeslice to end at")
    parser.add_argument("-a", "--align_to", type=str, default="first",
                        help="which model to align every other model to: 'first', 'last', or 'previous'")
    parser.add_argument("-c", "--compare_to", type=str, default="first",
                        help="which model's vector to compare every other model's vector to: 'first', 'last', or 'previous'")
    parser.add_argument("-m", "--distance_measure", type=str, default="cosine",
                        help="which distance measure to use 'cosine', or 'neighborhood'")
    parser.add_argument("-k", "--k_neighbors", type=int, default=25,
                        help="Number of neighbors to use for neighborhood shift distance measure")
    parser.add_argument("-s", "--n_samples", type=int, default=1000,
                        help="Number of samples to draw for permutation test")
    parser.add_argument("-p", "--p_value_threshold", type=float, default=0.05, help="P-value cut-off")
    parser.add_argument("-g", "--gamma_threshold", type=float, default=0, help="Minimum z-score magnitude.")
    parser.add_argument("-r", "--rank_by", type=str, default='p_value',
                        help="What to rank words by: 'p_value', 'z_score', or 'mean_shift'")
    parser.add_argument("-n", "--n_best", type=int, default=1000, help="Size of n-best list to store")
    parser.add_argument("-v", "--vocab_threshold", type=int, default=75,
                        help="percent of models which must contain word in order for it to be included")
    parser.add_argument("-o", "--outfiles_dir", type=str,
                        default="/results/change_point_candidates/nov_29/monthly/skipgram/continuous_aligned"
                                "/vec_200_w10_mc500_iter15/2012_01_to_2017_06/",
                        help="Path to file where results will be written")
    parser.add_argument("-vs", "--vector_size", type=int, default=128, help="vector size")
    parser.add_argument("-ws", "--window_size", type=int, default=5, help="window size")
    parser.add_argument("-mc", "--min_count", type=int, default=20, help="min count")
    parser.add_argument("-ni", "--no_of_iter", type=int, default=1, help="no of iteration")
    parser.add_argument("-lc", "--lower_case", type=int, default=0)
    parser.add_argument("-cl", "--clean", type=int, default=1)
    parser.add_argument("-t", "--training_mode", type=str, default='independent',
                        help="training mode: was it independent or continuous? -- if you want to use alignment, "
                             "say independent.")
    parser.add_argument("-z", "--z_scores", action="store_true", default=False,
                        help="Include this flag to standardize the distances (i.e. use z-scores). If this flag is not "
                             "included, the raw cosine or neighbourhood scores will be used without standardization.")
    parser.add_argument("-sg", "--skipgram", type=int, default=0, help="1 = skipgram, 0 = CBOW")
    parser.add_argument("--targets", type=str,
                        help="Path to a csv (or single column) file with target words. If present, only those will be cosidered")
    parser.add_argument("--glove", action="store_true", default=False)
    options = parser.parse_args()

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
    start_time = datetime.datetime.now()
    print("Starting at {}".format(start_time))

    # First, we construct a list of the filepaths of all the models we have, and a list of the time-slices they
    # correspond to. We initalize the vocab to the set of words which occur in at least v% of the models.

    model_paths = []
    time_slice_labels = []
    timeslices = sorted(
        [ts for ts in os.listdir(options.models_rootdir) if os.path.isdir(os.path.join(options.models_rootdir, ts))])

    vocab_filepath = "{}/time_series_vocab_{}pc_wc{}_{}_to_{}.txt".format(options.models_rootdir,
                                                                          options.vocab_threshold, options.min_count,
                                                                          timeslices[0], timeslices[-1])

    # (first_year, first_month) = (int(i) for i in options.first_timeslice.split('-'))
    # (last_year, last_month) = (int(i) for i in options.last_timeslice.split('-'))

    # if we've already stored the common vocab, can just read it, don't have to load all the models and check their
    # vocab
    if os.path.isfile(vocab_filepath):

        vocab = set()
        with open(vocab_filepath, 'r') as infile:
            for line in infile:
                vocab.add(line.strip())

        for ts in timeslices:
            if options.glove:
                model_path = "{}/{}/glove/vectors.txt".format(options.models_rootdir, ts)
            else:
                model_path = "{}/{}/vec_{}_w{}_mc{}_iter{}_sg{}/saved_model.gensim".format(
                    options.models_rootdir, ts, options.vector_size,
                    options.window_size,
                    options.min_count,
                    options.no_of_iter,
                    options.skipgram)
            print(model_path)
            if os.path.isfile(model_path):
                model_paths.append(model_path)
                time_slice_labels.append(ts)

        n_models = len(model_paths)

    else:
        # if we HAVEN'T already stored the common vocab, we DO need to load all the models and check their vocab

        vocab_counter = Counter()
        for ts in timeslices:
            print("I'M HERE")
            if options.glove:
                model_path = "{}/{}/glove/vectors.txt".format(options.models_rootdir, ts)
                print(model_path)
            else:
                model_path = "{}/{}/vec_{}_w{}_mc{}_iter{}_sg{}/saved_model.gensim".format(
                    options.models_rootdir, ts, options.vector_size,
                    options.window_size,
                    options.min_count,
                    options.no_of_iter,
                    options.skipgram)
            try:
                model = load_model(model_path)
            except FileNotFoundError:
                pass
            else:
                print("loaded {} at {}".format(ts, datetime.datetime.now()))
                model_paths.append(model_path)
                time_slice_labels.append(ts)
                vocab_counter.update(model.vocab.keys())

        n_models = len(model_paths)
        print(vocab_counter.most_common(10))
        vocab = set([w for w in vocab_counter if vocab_counter[w] >= options.vocab_threshold * 0.01 * n_models])
        del vocab_counter

        with open(vocab_filepath, 'w') as outfile:
            for word in vocab:
                outfile.write(word + '\n')

    print("\nGot vocab at {}".format(datetime.datetime.now()))
    print("size of vocab: {}\n".format(len(vocab)))

    print("\n\nMODEL PATHS:")
    print(model_paths)

    print("\n\nTIME SLICE LABELS:")
    print(time_slice_labels)

    print("\n\nfound {} models.".format(n_models))

    distances_filepath = options.outfiles_dir + '/time_series_analysis_distances_f{}_l{}_a{}_c{}_m{}_k{}_v{}.json'.format(
        timeslices[0], timeslices[-1], options.align_to, options.compare_to, options.distance_measure,
        options.k_neighbors, options.vocab_threshold)
    zscores_filepath = options.outfiles_dir + '/time_series_analysis_z_scores_f{}_l{}_a{}_c{}_m{}_k{}_v{}.json'.format(
        timeslices[0], timeslices[-1], options.align_to, options.compare_to, options.distance_measure,
        options.k_neighbors, options.vocab_threshold)

    # if we've already computed the distances, can just read them.
    if os.path.isfile(distances_filepath) and os.path.isfile(zscores_filepath):
        with open(distances_filepath, 'r') as infile:
            dict_of_dist_dicts = json.load(infile)
        with open(zscores_filepath, 'r') as infile:
            dict_of_z_score_dicts = json.load(infile)
        time_slices_used = sorted(list(dict_of_dist_dicts.keys()))
        print("\n\nLIST OF TIME SLICES USED:")
        print(time_slices_used)

    else:

        pool1 = multiprocessing.Pool(12)

        dist_dicts = pool1.map(get_dist_dict_multithreaded, enumerate(model_paths))

        dict_of_dist_dicts = {}
        for tup in dist_dicts:
            if tup:
                (i, dist_dict) = tup
                dict_of_dist_dicts[time_slice_labels[i]] = dist_dict

        time_slices_used = sorted(list(dict_of_dist_dicts.keys()))
        print("\n\nLIST OF TIME SLICES USED:")
        print(time_slices_used)

        dict_of_z_score_dicts = {}
        for t in time_slice_labels:
            if t in dict_of_dist_dicts:
                dict_of_z_score_dicts[t] = get_z_score_dict(dict_of_dist_dicts[t])

        os.makedirs(options.outfiles_dir, exist_ok=True)
        with open(distances_filepath, 'w') as outfile:
            json.dump(dict_of_dist_dicts, outfile, indent=2)

        with open(zscores_filepath, 'w') as outfile:
            json.dump(dict_of_z_score_dicts, outfile, indent=2)

    print("GOT DICTS OF DIST AND Z-SCORE DICTS at {}\n".format(datetime.datetime.now()))

    # Finally, we do the change-point analysis on each word's z-score (or dist) time-series. We keep a ranked list of
    # the n 'best' change-points detected, and print it when we're done.

    pool2 = multiprocessing.Pool(12)

    if options.z_scores:

        dict_of_z_scores_by_word = defaultdict(lambda: defaultdict(list))
        for word in vocab:
            for time_slice in dict_of_z_score_dicts:
                if word in dict_of_z_score_dicts[time_slice]:
                    dict_of_z_scores_by_word[word][time_slice].append(dict_of_z_score_dicts[time_slice][word])

        results = pool2.map(get_word_change_point, dict_of_z_scores_by_word.items())


    else:

        dict_of_dists_by_word = defaultdict(lambda: defaultdict(list))
        for word in vocab:
            for time_slice in dict_of_dist_dicts:
                if word in dict_of_dist_dicts[time_slice]:
                    dict_of_dists_by_word[word][time_slice].append(dict_of_dist_dicts[time_slice][word])

        results = pool2.map(get_word_change_point, dict_of_dists_by_word.items())

    print('got {} results'.format(len(results)))
    # columns for tsv file: 0 = word, 1 = timestep, 2 = p-value, 3 = mean-shift, 4 = z_score
    results = [r for r in results if r]
    print('got {} not-none results'.format(len(results)))

    if options.rank_by == 'z_score':
        results = sorted(results, key=lambda x: -x[4])
    elif options.rank_by == 'mean_shift':
        results = sorted(results, key=lambda x: -x[3])
    else:  # options.rank_by == 'p_value'
        # we'll actually rank by mean-shift first and then p-value, so that words with the same p-value are sorted by
        # the size of the mean-shift.
        results = sorted(results, key=lambda x: -x[3])
        results = sorted(results, key=lambda x: x[2])
    # else: raise RunTimeError("Invalid command line argument: Only possible values for option -r (--rank_by) are
    # 'z_score', 'mean_shift', or 'p_value'")

    os.makedirs(options.outfiles_dir, exist_ok=True)
    if options.z_scores:
        outfile_path = options.outfiles_dir + '/time_series_analysis_standardized_output_f{}_l{}_a{}_c{}_m{}_k{}_s{}_p{}_g{}_v{}.tsv'.format(
            timeslices[0], timeslices[-1], options.align_to, options.compare_to,
            options.distance_measure, options.k_neighbors, options.n_samples, options.p_value_threshold,
            options.gamma_threshold, options.vocab_threshold)
    else:
        outfile_path = options.outfiles_dir + '/time_series_analysis_NOT_standardized_output_f{}_l{}_a{}_c{}_m{}_k{}_s{}_p{}_g{}_v{}.tsv'.format(
            timeslices[0], timeslices[-1], options.align_to, options.compare_to,
            options.distance_measure, options.k_neighbors, options.n_samples, options.p_value_threshold,
            options.gamma_threshold, options.vocab_threshold)

    # columns for tsv file: 0 = word, 1 = timestep, 2 = p-value, 3 = mean-shift, 4 = z_score
    with open(outfile_path, 'w') as outfile:
        # outfile.write("\t".join(["word", "timestep", "p-value", "mean-shift", "z_score"]) + "\n")
        for (i, item) in enumerate(results):
            word = str(item[0])

            # Filter words that are not in the optional target list
            if len(targets) != 0 and word not in targets:
                continue

            # If no target list is are specified, break after options.n_best
            if len(targets) == 0 and i > options.n_best:
                break

            # print(i, ":", item)
            outfile.write('\t'.join([str(s) for s in item]) + '\n')

    print("All done at {}. Writing log file...\n".format(datetime.datetime.now()))
    write_logfile(outfile_path, options, start_time)
    print("Written log file.")

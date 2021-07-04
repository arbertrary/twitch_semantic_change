import argparse
# import multiprocessing
import os
from collections import Counter, defaultdict
import torch
import datetime
import json
from scipy.spatial.distance import cosine
import numpy as np
import itertools

import twopoint_lsc as tp
from torch.multiprocessing import Pool, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def get_dist_dict_multithreaded(tup):
    (i, model_path, compare_to, align_to, vocab) = tup
    if i == 0 and compare_to == "first":
        return None
    elif i == len(model_paths) - 1 and compare_to == "last":
        return None
    else:
        if align_to == "first":
            alignment_reference_model_path = model_paths[0]
        else:  # align to last
            alignment_reference_model_path = model_paths[-1]

        if compare_to == 'first':
            comparison_reference_model_path = model_paths[0]
        else:  # compare to last
            comparison_reference_model_path = model_paths[-1]

        dist_dict = get_dist_dict(model_path, alignment_reference_model_path, comparison_reference_model_path, vocab)

        return (i, dist_dict)


def get_dist_dict(model_path, alignment_reference_model_path, comparison_reference_model_path, common_vocab):
    model = torch.load(model_path)
    # alignment_reference_model = torch.load(alignment_reference_model_path)
    comparison_reference_model = torch.load(comparison_reference_model_path)

    # if alignment_reference_model_path != comparison_reference_model_path:
    #     _, _, comparison_reference_matrix = tp.align_models(alignment_reference_model, comparison_reference_model)
    #     _, _, model_matrix = tp.align_models(alignment_reference_model, model)
    # else:
    intersection_vocab, comparison_reference_matrix, model_matrix = tp.align_models(comparison_reference_model,
                                                                                    model)
    dist_dict = {}

    for word in common_vocab:
        if word in intersection_vocab:
            index = intersection_vocab.index(word)
            dist_dict[word] = cosine(comparison_reference_matrix[index], model_matrix[index])

    return dist_dict


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
    (word, dists_dict, n_samples, p_value_threshold, gamma_threshold, compare_to) = tup

    change_point = detect_change_point(word, dists_dict, n_samples, p_value_threshold,
                                       gamma_threshold, compare_to)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_rootdir", type=str,
                        help="path to first embedding model")
    parser.add_argument("-a", "--align_to", type=str, default="first",
                        help="which model to align every other model to: 'first', 'last', or 'previous'")
    parser.add_argument("-c", "--compare_to", type=str, default="first",
                        help="which model's vector to compare every other model's vector to: 'first', 'last', or 'previous'")
    parser.add_argument("-v", "--vocab_threshold", type=int, default=75,
                        help="percent of models which must contain word in order for it to be included")
    parser.add_argument("-t", "--t_best", type=int, default=1000, help="Number of top-ranked words to output")
    parser.add_argument("-o", "--outfiles_dir", type=str,
                        help="Path to file where results will be written")
    parser.add_argument("-s", "--n_samples", type=int, default=100,
                        help="Number of samples to draw for permutation test")
    parser.add_argument("-p", "--p_value_threshold", type=float, default=0.05, help="P-value cut-off")
    parser.add_argument("-g", "--gamma_threshold", type=float, default=0, help="Minimum z-score magnitude.")
    parser.add_argument("-z", "--z_scores", action="store_true", default=True,
                        help="Include this flag to standardize the distances (i.e. use z-scores). If this flag is not "
                             "included, the raw cosine or neighbourhood scores will be used without standardization.")

    options = parser.parse_args()

    start_time = datetime.datetime.now()
    print("Starting at {}".format(start_time))

    # align_to = options.align_to
    # compare_to = options.compare_to
    model_paths = []
    time_slice_labels = []
    timeslices = sorted([ts for ts in os.listdir(options.models_rootdir)])

    os.makedirs(options.outfiles_dir, exist_ok=True)
    vocab_filepath = "{}/time_series_vocab_{}pc_{}_to_{}.txt".format(options.outfiles_dir, options.vocab_threshold,
                                                                     timeslices[0], timeslices[-1])

    vocab_counter = Counter()
    for ts in timeslices:
        model_path = "{}/{}/fused_vectors.pt".format(options.models_rootdir, ts)
        model = torch.load(model_path)

        model_paths.append(model_path)
        time_slice_labels.append(ts)
        vocab_counter.update(model.keys())

    n_models = len(model_paths)
    print(vocab_counter.most_common(10))
    vocab = set([w for w in vocab_counter if vocab_counter[w] >= options.vocab_threshold * 0.01 * n_models])
    del vocab_counter

    with open(vocab_filepath, 'w', encoding="utf-8") as outfile:
        for word in vocab:
            outfile.write(word + '\n')

    print("\nGot vocab at {}".format(datetime.datetime.now()))
    print("size of vocab: {}\n".format(len(vocab)))

    print("\n\nMODEL PATHS:")
    print(model_paths)

    print("\n\nTIME SLICE LABELS:")
    print(time_slice_labels)

    print("\n\nfound {} models.".format(n_models))

    distances_filepath = options.outfiles_dir + '/time_series_analysis_distances_f{}_l{}_a{}_c{}_v{}.json'.format(
        timeslices[0], timeslices[-1], options.align_to, options.compare_to, options.vocab_threshold)
    zscores_filepath = options.outfiles_dir + '/time_series_analysis_z_scores_f{}_l{}_a{}_c{}_v{}.json'.format(
        timeslices[0], timeslices[-1], options.align_to, options.compare_to, options.vocab_threshold)

    # if os.path.isfile(distances_filepath) and os.path.isfile(zscores_filepath):
    if os.path.isfile(distances_filepath) and os.path.isfile(zscores_filepath):
        with open(distances_filepath, 'r', encoding="utf-8") as infile:
            dict_of_dist_dicts = json.load(infile)
        with open(zscores_filepath, 'r', encoding="utf-8") as infile:
            dict_of_z_score_dicts = json.load(infile)
        time_slices_used = sorted(list(dict_of_dist_dicts.keys()))
        print("\n\nLIST OF TIME SLICES USED:")
        print(time_slices_used)

    else:
        # print("STOP BEFORE THE DIST DICTS ARE CALCULATED AGAIN")
        # exit()

        pool1 = Pool(10)
        inputs = [tup + (options.compare_to, options.align_to, vocab) for tup in enumerate(model_paths)]
        dist_dicts = pool1.map(get_dist_dict_multithreaded, inputs)
        # dist_dicts = [get_dist_dict_multithreaded(tup) for tup in enumerate(model_paths)]

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

        with open(distances_filepath, 'w', encoding="utf-8") as outfile:
            json.dump(dict_of_dist_dicts, outfile, indent=2)

        with open(zscores_filepath, 'w') as outfile:
            json.dump(dict_of_z_score_dicts, outfile, indent=2)

    print("GOT DICTS OF DIST AND Z-SCORE DICTS at {}\n".format(datetime.datetime.now()))

    pool2 = Pool(10)
    if options.z_scores:

        dict_of_z_scores_by_word = defaultdict(lambda: defaultdict(list))
        for word in vocab:
            for time_slice in dict_of_z_score_dicts:
                if word in dict_of_z_score_dicts[time_slice]:
                    dict_of_z_scores_by_word[word][time_slice].append(dict_of_z_score_dicts[time_slice][word])

        inputs = [tup + (options.n_samples, options.p_value_threshold, options.gamma_threshold, options.compare_to) for
                  tup in dict_of_z_scores_by_word.items()]
        results = pool2.map(get_word_change_point, inputs)
        # results = [get_word_change_point(x) for x in dict_of_z_scores_by_word.items()]

    else:
        dict_of_dists_by_word = defaultdict(lambda: defaultdict(list))
        for word in vocab:
            for time_slice in dict_of_dist_dicts:
                if word in dict_of_dist_dicts[time_slice]:
                    dict_of_dists_by_word[word][time_slice].append(dict_of_dist_dicts[time_slice][word])
        inputs = [tup + (options.n_samples, options.p_value_threshold, options.gamma_threshold, options.compare_to) for
                  tup in dict_of_dists_by_word.items()]
        results = pool2.map(get_word_change_point, inputs)

        # results = [get_word_change_point(x) for x in dict_of_dists_by_word.items()]

    print('got {} results'.format(len(results)))
    # columns for tsv file: 0 = word, 1 = timestep, 2 = p-value, 3 = mean-shift, 4 = z_score
    results = [r for r in results if r]
    print('got {} not-none results'.format(len(results)))

    results = sorted(results, key=lambda x: -x[3])
    results = sorted(results, key=lambda x: x[2])

    if options.z_scores:
        outfile_path = options.outfiles_dir + '/time_series_analysis_standardized_output_f{}_l{}_a{}_c{}_s{}_v{}.tsv'.format(
            timeslices[0], timeslices[-1], options.align_to, options.compare_to, options.n_samples,
            options.vocab_threshold)
    else:
        outfile_path = options.outfiles_dir + '/time_series_analysis_NOT_standardized_output_f{}_l{}_a{}_c{}_s{}_v{}.tsv'.format(
            timeslices[0], timeslices[-1], options.align_to, options.compare_to, options.n_samples,
            options.vocab_threshold)

    with open(outfile_path, 'w', encoding="utf-8") as outfile:
        # outfile.write("\t".join(["word", "timestep", "p-value", "mean-shift", "z_score"]) + "\n")
        for (i, item) in enumerate(results):
            word = str(item[0])

            # Filter words that are not in the optional target list
            # if len(targets) != 0 and word not in targets:
            #     continue
            #
            # # If no target list is are specified, break after options.n_best
            if i > options.n_best:
                break

            # print(i, ":", item)
            outfile.write('\t'.join([str(s) for s in item]) + '\n')

    print("All done at {}. Writing log file...\n".format(datetime.datetime.now()))
    write_logfile(outfile_path, options, start_time)
    print("Written log file.")

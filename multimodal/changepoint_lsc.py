import argparse
#import multiprocessing
import os
from collections import Counter
import torch
import datetime
import json
from scipy.spatial.distance import cosine

import twopoint_lsc as tp
from torch.multiprocessing import Pool, set_start_method


def get_dist_dict_multithreaded(tup):
    (i, model_path) = tup
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
    parser.add_argument("-t", "--t_best", type=int, default=20, help="Number of top-ranked words to output")
    parser.add_argument("-o", "--outfiles_dir", type=str,
                        help="Path to file where results will be written")
    options = parser.parse_args()
    align_to = options.align_to
    compare_to = options.compare_to
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

    with open(vocab_filepath, 'w',encoding="utf-8") as outfile:
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

    if os.path.isfile(distances_filepath) and os.path.isfile(zscores_filepath):
        with open(distances_filepath, 'r',encoding="utf-8") as infile:
            dict_of_dist_dicts = json.load(infile)
        with open(zscores_filepath, 'r',encoding="utf-8") as infile:
            dict_of_z_score_dicts = json.load(infile)
        time_slices_used = sorted(list(dict_of_dist_dicts.keys()))
        print("\n\nLIST OF TIME SLICES USED:")
        print(time_slices_used)

    else:
        
        try:
                 set_start_method('spawn')
        except RuntimeError:
                pass
            

        #pool1 = Pool(10)
        #dist_dicts = pool1.map(get_dist_dict_multithreaded, enumerate(model_paths))
        dist_dicts = [get_dist_dict_multithreaded(tup) for tup in enumerate(model_paths)]

        dict_of_dist_dicts = {}
        for tup in dist_dicts:
            if tup:
                (i, dist_dict) = tup
                dict_of_dist_dicts[time_slice_labels[i]] = dist_dict
        time_slices_used = sorted(list(dict_of_dist_dicts.keys()))
        print("\n\nLIST OF TIME SLICES USED:")
        print(time_slices_used)

        with open(distances_filepath, 'w',encoding="utf-8") as outfile:
            json.dump(dict_of_dist_dicts, outfile, indent=2)

        # with open(zscores_filepath, 'w') as outfile:
        #     json.dump(dict_of_z_score_dicts, outfile, indent=2)

    print("GOT DICTS OF DIST AND Z-SCORE DICTS at {}\n".format(datetime.datetime.now()))

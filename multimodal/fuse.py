import argparse

import gensim.models
import torch
from torch import nn
import json
from ast import literal_eval as make_tuple
import numpy as np

CONFIG = {"latent_dim": 128}


class AutoFusion(nn.Module):
    """
    This Code is taken from the repository https://github.com/Demfier/philo
    Which implements the paper "Adaptive Fusion Techniques for Multimodal Data"
    https://arxiv.org/abs/1911.03821
    """

    def __init__(self, config, input_features):
        super(AutoFusion, self).__init__()
        self.config = config
        self.input_features = input_features

        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, input_features // 2),
            nn.Tanh(),
            nn.Linear(input_features // 2, config['latent_dim']),
            nn.ReLU()
        )
        self.fuse_out = nn.Sequential(
            nn.Linear(config['latent_dim'], input_features // 2),
            nn.ReLU(),
            nn.Linear(input_features // 2, input_features)
        )
        self.criterion = nn.MSELoss()

    def forward(self, z):
        compressed_z = self.fuse_in(z)
        loss = self.criterion(self.fuse_out(compressed_z), z)
        output = {
            'z': compressed_z,
            'loss': loss
        }
        return output


def load_gensim_model(model_path):
    m = gensim.models.Word2Vec.load(model_path)
    m = m.wv
    m.init_sims(replace=True)
    return m


def load_vocab(vocab_path):
    v = []
    with open(vocab_path, "r") as jsonfile:
        vocab = json.loads(jsonfile.read())
        for w in vocab:
            split = make_tuple(w)
            tp = (split[0], split[1:])
            v.append(tp)

    return v


if __name__ == '__main__':
    vocab = load_vocab("vocab.json")
    print(vocab)
    exit()

    parser = argparse.ArgumentParser(description="generating an embedding")
    parser.add_argument("-e", "--emote_model", type=str, help="Path to the emote word embedding model")
    parser.add_argument("-w", "--word_model", type=str, help="Path to the word embedding model")
    parser.add_argument("-v", "--vocab", type=str, help="Path to the multimodal vocabulary")
    parser.add_argument("--epochs", type=int)

    args = vars(parser.parse_args())

    emote_model_path = args["emote_model"]
    word_model_path = args["word_model"]
    vocab_path = args["vocab"]

    vocabulary = load_vocab(vocab_path)
    emote_model = load_gensim_model(emote_model_path)
    word_model = load_gensim_model(word_model_path)
    # Plan:
    # Embedding vektoren laden, mit
    # torch.tensor() zu tensor machen
    # fused vektoren speichern in dictionary
    # word: tensor
    model = AutoFusion(CONFIG, 128 * 2)
    for w in vocab:
        word = w[0]
        emotes = w[1]

        word_vector = word_model[word]
        # for the case of images possibly only this line needs to be changed?
        emote_vector = np.mean([emote_model[emote] for emote in emotes], axis=1)
        w_input = torch.tensor(word_vector)
        e_input = torch.tensor(emote_vector)
        input_concat = torch.cat([w_input, e_input])
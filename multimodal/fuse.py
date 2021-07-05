import argparse

import gensim.models
import torch
from torch import optim
from torch import nn
import json
from ast import literal_eval as make_tuple
import numpy as np
import os

CONFIG = {
    "latent_dim": 128,
    "lr": 1e-3,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"  # gpu_id ('x' => multiGPU)
}


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
    # Vectors are normalized here, that means vectors_norm == m
    m.init_sims(replace=True)
    return m


def load_tuple_vocab(vocab_path):
    with open(vocab_path, "r") as jsonfile:
        vocab = json.loads(jsonfile.read())

        vocab_tuples = []

        for word_emote_tuple in vocab:
            split = word_emote_tuple.split("|")
            word = split[0]
            emotes = list(split[1:])
            vocab_tuples.append((word, emotes))

        print(vocab_tuples)
        return vocab_tuples


def load_vocab(vocab_path):
    with open(vocab_path, "r") as jsonfile:
        vocab = json.loads(jsonfile.read())

        vocab_tuples = []
        for word in vocab:
            print(word)
            emotes = list(vocab[word]["emotes"].keys())
            vocab_tuples.append((word, emotes))

        print(vocab_tuples)
        return vocab_tuples


def get_input_tensors(word_model, emote_model, vocab):
    inputs = []
    for word_emote_tuple in vocab:
        (word, emotes) = word_emote_tuple

        if word not in word_model:
            word = "UNK"
            # word_emote_tuple = word_emote_tuple.replace(word, "UNK")
            word_vector = torch.zeros(CONFIG["latent_dim"])
        else:
            word_vector = torch.tensor(word_model[word])

        # print("word_vector", len(word_vector))

        # for the case of images possibly only this part needs to be changed?
        if len(emotes) == 1:
            if emotes[0] not in emote_model:
                emotes[0] = "UNK_EM"
                # word_emote_tuple = word_emote_tuple.replace(emotes[0], "UNK_EM")
                emote_vector = torch.zeros(CONFIG["latent_dim"])
            else:
                if args["images"]:
                    emote_vector = emote_model[emotes[0]]
                else:
                    emote_vector = torch.tensor(emote_model[emotes[0]])
        else:
            vectors = []
            for emote in emotes:
                if emote in emote_model:
                    vectors.append(emote_model[emote])
                else:
                    word_emote_tuple = word_emote_tuple.replace(emote, "UNK_EM")

            # vectors = [emote_model[emote] for emote in emotes if emote in emote_model]
            if len(vectors) == 1:
                if args["images"]:
                    emote_vector = vectors[0]
                else:
                    emote_vector = torch.tensor(vectors[0])
            elif len(vectors) == 0:
                emote_vector = torch.zeros(CONFIG["latent_dim"])
            else:
                if args["images"]:
                    stacked = torch.stack(vectors)
                    emote_vector = torch.mean(stacked, dim=0)
                else:
                    emote_vector = torch.tensor(np.mean(vectors, axis=0))

        word_vector = word_vector.to(CONFIG["device"])
        emote_vector = emote_vector.to(CONFIG["device"])
        input_concat = torch.cat([word_vector, emote_vector])
        input_concat = input_concat.to(CONFIG["device"])
        # print(input_concat.shape)

        if args["tuples"]:
            key = "|".join([word] + emotes)
            inputs.append((key, input_concat))
        else:
            inputs.append((word, input_concat))

    return inputs


if __name__ == '__main__':
    # Local testrun command:
    # python multimodal/fuse.py
    # --emote_model=data/testdata/testmodels/emote_model/saved_model.gensim
    # --word_model=data/testdata/testmodels/word_model/saved_model.gensim
    # --vocab=multimodal/vocab.json
    # --out_dir=data/testdata/testmodels/
    # --tuples
    parser = argparse.ArgumentParser(description="generating an embedding")
    parser.add_argument("-e", "--emote_model", type=str, help="Path to the emote word embedding model")
    parser.add_argument("-w", "--word_model", type=str, help="Path to the word embedding model")
    parser.add_argument("-v", "--vocab", type=str, help="Path to the multimodal vocabulary")
    parser.add_argument("-o", "--out_dir", type=str, help="Path to the save location for the autofused tensors")
    parser.add_argument("-im", "--images", action="store_true", default=False)
    parser.add_argument("-t", "--tuples", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=10)

    args = vars(parser.parse_args())

    emote_model_path = args["emote_model"]
    word_model_path = args["word_model"]
    vocab_path = args["vocab"]
    out_dir = args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args["tuples"]:
        vocabulary = load_tuple_vocab(vocab_path)
    else:
        vocabulary = load_vocab(vocab_path)

    if args["images"]:
        emote_model = torch.load(emote_model_path)
    else:
        emote_model = load_gensim_model(emote_model_path)

    word_model = load_gensim_model(word_model_path)
    # Plan:
    # Embedding vektoren laden, mit
    # torch.tensor() zu tensor machen
    # fused vektoren speichern in dictionary
    # word: tensor
    device = torch.device(CONFIG["device"])
    model = AutoFusion(CONFIG, CONFIG["latent_dim"] * 2)
    model = model.to(device)

    # TODO
    # Cuda/GPU nutzen? Bringt das hier überhaupt was?

    # TODO
    # logging

    vocabulary = load_vocab(vocab_path)
    inputs = get_input_tensors(word_model, emote_model, vocabulary)
    out_tensors = {}

    optimizer = optim.Adam(model.parameters(), CONFIG["lr"])
    for epoch in range(args["epochs"]):
        epoch_loss = []
        for w, tensor in inputs:
            optimizer.zero_grad()
            output = model(tensor)

            out_tensors[w] = output["z"]
            loss = output["loss"]

            loss.backward()
            optimizer.step()
            epoch_loss.append(output["loss"].item())

        print(np.mean(epoch_loss))

    torch.save(out_tensors, os.path.join(out_dir, "fused_vectors.pt"))

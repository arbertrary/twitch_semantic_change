import argparse

import gensim.models
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import datetime

from torch.multiprocessing import Pool, set_start_method

try:
        set_start_method('spawn')
except RuntimeError:
        pass



class FuseDataset(Dataset):
    def __init__(self, word_model, emote_model, vocab, images: bool, tuples: bool,config):
        """
        Args:
        """
        self.word_model = word_model
        self.emote_model = emote_model
        self.vocab = vocab
        self.tuples = tuples
        self.images = images
        self.config = config

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        (word, emotes) = self.vocab[idx]

        if word not in self.word_model:
            word = "UNK"
            # word_emote_tuple = word_emote_tuple.replace(word, "UNK")
            word_vector = torch.zeros(self.config["latent_dim"])
        else:
            word_vector = torch.tensor(self.word_model[word])

        # for the case of images possibly only this part needs to be changed?
        if len(emotes) == 1:
            if emotes[0] not in self.emote_model:
                emotes[0] = "UNK_EM"
                # word_emote_tuple = word_emote_tuple.replace(emotes[0], "UNK_EM")
                emote_vector = torch.zeros(self.config["latent_dim"])
            else:
                if self.images:
                    emote_vector = self.emote_model[emotes[0]]
                else:
                    emote_vector = torch.tensor(self.emote_model[emotes[0]])
        else:
            vectors = []
            temp_emotes = emotes
            for i, emote in enumerate(temp_emotes):
                if emote in self.emote_model:
                    vectors.append(self.emote_model[emote])
                else:
                    emotes[i] = "UNK_EM"
                    # word_emote_tuple = word_emote_tuple.replace(emote, "UNK_EM")

            # vectors = [emote_model[emote] for emote in emotes if emote in emote_model]
            if len(vectors) == 1:
                if self.images:
                    emote_vector = vectors[0]
                else:
                    emote_vector = torch.tensor(vectors[0])
            elif len(vectors) == 0:
                emote_vector = torch.zeros(self.config["latent_dim"])
            else:
                if self.images:
                    stacked = torch.stack(vectors)
                    emote_vector = torch.mean(stacked, dim=0)
                else:
                    emote_vector = torch.tensor(np.mean(vectors, axis=0))

        word_vector = word_vector.to(self.config["device"], non_blocking=True)
        emote_vector = emote_vector.to(self.config["device"], non_blocking=True)
        input_concat = torch.cat([word_vector, emote_vector])
        input_concat = input_concat.to(self.config["device"], non_blocking=True)
        # print(input_concat.shape)

        if self.tuples:
            key = "|".join([word] + emotes)
            sample = {key: input_concat}
            # inputs.append((key, input_concat))
        else:
            key = word
            sample = {word: input_concat}
            # inputs.append((word, input_concat))

        # print(sample)

        return key, input_concat


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
            nn.Linear(input_features // 2, self.config['latent_dim']),
            nn.ReLU()
        )
        self.fuse_out = nn.Sequential(
            nn.Linear(self.config['latent_dim'], input_features // 2),
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
    with open(vocab_path, "r", encoding="utf-8") as jsonfile:
        vocab = json.loads(jsonfile.read())

        vocab_tuples = []

        for word_emote_tuple in vocab:
            split = word_emote_tuple.split("|")
            word = split[0]
            emotes = list(split[1:])
            vocab_tuples.append((word, emotes))

        # print(vocab_tuples)
        return vocab_tuples


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as jsonfile:
        vocab = json.loads(jsonfile.read())

        vocab_tuples = []
        for word in vocab:
            emotes = list(vocab[word]["emotes"].keys())
            vocab_tuples.append((word, emotes))

        return vocab_tuples


if __name__ == '__main__':
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
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--gpu", action="store_true", default=False)

    args = vars(parser.parse_args())
    CONFIG = {
        "latent_dim": 128,
        "lr": 1e-3,
        "device": "cuda:0" if args["gpu"] else "cpu"  # gpu_id ('x' => multiGPU)
    }
    print("CONFIG:")
    print(CONFIG)

    start_time = datetime.datetime.now()
    print("Starting at {}".format(start_time))

    emote_model_path = args["emote_model"]
    word_model_path = args["word_model"]
    vocab_path = args["vocab"]
    out_dir = args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args["tuples"]:
        vocabulary = load_tuple_vocab(vocab_path)
    else:
        vocabulary = load_vocab(vocab_path)

    print("\nGot vocab at {}".format(datetime.datetime.now()))

    if args["images"]:
        emote_model = torch.load(emote_model_path)
    else:
        emote_model = load_gensim_model(emote_model_path)

    word_model = load_gensim_model(word_model_path)

    print("\nLoaded models at {}".format(datetime.datetime.now()))

    device = torch.device(CONFIG["device"])
    model = AutoFusion(CONFIG, CONFIG["latent_dim"] * 2)
    model = model.to(device)
    model.train()

    # TODO
    # logging

    # inputs = get_input_tensors(word_model, emote_model, vocabulary)
    inputs = FuseDataset(word_model=word_model, emote_model=emote_model, vocab=vocabulary, images=args["images"],
                         tuples=args["tuples"],config=CONFIG)

    dataloader = DataLoader(inputs, batch_size=512, shuffle=True, num_workers=6)

    out_tensors = {}

    print("\nGot inputs at {}".format(datetime.datetime.now()))

    optimizer = optim.Adam(model.parameters(), CONFIG["lr"])
    for epoch in range(args["epochs"]):
        epoch_loss = []
        # for w, tensor in inputs:
        for i_batch, (w, tensor) in enumerate(dataloader):
            # print(w, tensor)
            output = model(tensor)

            out_tensors[w] = output["z"]
            loss = output["loss"]

            loss.backward()
            optimizer.step()
            epoch_loss.append(output["loss"].item())
            optimizer.zero_grad()

        print("\nEpoch {} done at {}".format(epoch, datetime.datetime.now()))
        print(np.mean(epoch_loss))

    torch.save(out_tensors, os.path.join(out_dir, "fused_vectors.pt"))

    print("\nSaving done at {}".format(datetime.datetime.now()))
    end_time = datetime.datetime.now()
    print("\nExecution took {} Seconds".format((end_time - start_time).total_seconds()))

import argparse

import gensim.models
import torch
from torch import nn
import json
from ast import literal_eval as make_tuple
import numpy as np

CONFIG = {
    "latent_dim": 128,
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
    m.init_sims(replace=True)
    return m


def load_vocab(vocab_path):
    v = []
    with open(vocab_path, "r") as jsonfile:
        vocab = json.loads(jsonfile.read())
    return vocab


if __name__ == '__main__':
    # vocab = load_vocab("vocab.json")
    # print(vocab)
    # exit()

    parser = argparse.ArgumentParser(description="generating an embedding")
    parser.add_argument("-e", "--emote_model", type=str, help="Path to the emote word embedding model")
    parser.add_argument("-w", "--word_model", type=str, help="Path to the word embedding model")
    parser.add_argument("-v", "--vocab", type=str, help="Path to the multimodal vocabulary")
    parser.add_argument("-o", "--out_path", type=str, help="Path to the save location for the autofused tensors")
    parser.add_argument("--epochs", type=int)

    args = vars(parser.parse_args())

    emote_model_path = args["emote_model"]
    word_model_path = args["word_model"]
    vocab_path = args["vocab"]
    out_path = args["out_path"]

    vocabulary = load_vocab(vocab_path)
    emote_model = load_gensim_model(emote_model_path)
    word_model = load_gensim_model(word_model_path)
    # Plan:
    # Embedding vektoren laden, mit
    # torch.tensor() zu tensor machen
    # fused vektoren speichern in dictionary
    # word: tensor
    device = CONFIG["device"]
    model = AutoFusion(CONFIG, CONFIG["latent_dim"] * 2)
    model = model.to(device)

    # TODO
    # Wie mach ich die Epochen? Hier noch ein
    # for i in range(epochs): drumherum?
    # Muss dann der output nochmal wieder reingeworfen werden?
    # Oder gibt's da eh nur einen einzigen Pass?
    # Den output muss ich dann irgendwo speichern als
    # {w: output["z"]}
    # z.B. {"("typeD4_pseudoword_bin2", "LUL")" : [1,2,3,4,5,...]}
    # Dann kann ich zwischen diesen vektoren die cos-distance berechnen
    # und dann mal schauen was rauskommt

    # TODO
    # Wo werf ich cuda rein?

    # TODO
    # Wie speichere ich die keyedvectors dann ab?

    # TODO
    # logging

    tensors = {}
    for w in vocabulary:
        split = make_tuple(w)
        word = split[0]
        print(w)
        print(word)
        emotes = list(split[1:])
        print(emotes)
        word_vector = word_model[word]
        print("word_vector", len(word_vector))
        # for the case of images possibly only this line needs to be changed?
        if len(emotes) == 1:
            emote_vector = emote_model[emotes[0]]
        else:
            vectors = [emote_model[emote] for emote in emotes]
            # print(vectors)
            emote_vector = np.mean([emote_model[emote] for emote in emotes], axis=0)
            # print(emote_vector)
        print("emote_vector", len(emote_vector))
        w_input = torch.tensor(word_vector)
        print("word_tensor", w_input.shape)
        e_input = torch.tensor(emote_vector)
        print("emote_tensor", e_input.shape)

        input_concat = torch.cat([w_input, e_input])
        print(input_concat.shape)

        output = model(input_concat)
        print(output["z"].shape)
        print("######\n")
        tensors[w] = output["z"]

    torch.save(tensors, out_path)
    # loaded = torch.load("multimodal/testfile")
    # print("### LOAD")
    # print(loaded["('OMEGALUL', 'OMEGALUL')"])

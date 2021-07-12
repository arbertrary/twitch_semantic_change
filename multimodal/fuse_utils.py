from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

CONFIG = {
    "latent_dim": 128,
    "lr": 1e-3,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"  # gpu_id ('x' => multiGPU)
}


class FuseDataset(Dataset):
    def __init__(self, word_model, emote_model, vocab, images: bool, tuples: bool):
        """
        Args:
        """
        self.word_model = word_model
        self.emote_model = emote_model
        self.vocab = vocab
        self.tuples = tuples
        self.images = images

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        (word, emotes) = self.vocab[idx]

        if word not in self.word_model:
            word = "UNK"
            # word_emote_tuple = word_emote_tuple.replace(word, "UNK")
            word_vector = torch.zeros(CONFIG["latent_dim"])
        else:
            word_vector = torch.tensor(self.word_model[word])

        # for the case of images possibly only this part needs to be changed?
        if len(emotes) == 1:
            if emotes[0] not in self.emote_model:
                emotes[0] = "UNK_EM"
                # word_emote_tuple = word_emote_tuple.replace(emotes[0], "UNK_EM")
                emote_vector = torch.zeros(CONFIG["latent_dim"])
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
                emote_vector = torch.zeros(CONFIG["latent_dim"])
            else:
                if self.images:
                    stacked = torch.stack(vectors)
                    emote_vector = torch.mean(stacked, dim=0)
                else:
                    emote_vector = torch.tensor(np.mean(vectors, axis=0))

        word_vector = word_vector.to(CONFIG["device"], non_blocking=True)
        emote_vector = emote_vector.to(CONFIG["device"], non_blocking=True)
        input_concat = torch.cat([word_vector, emote_vector])
        input_concat = input_concat.to(CONFIG["device"], non_blocking=True)
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

import sys
sys.path.append("../odenet")

from odenet import *

if __name__ == '__main__':

    myword = "zeigen"
    (lemma_id, lemma_value, pos, senses) = check_word_lemma(myword)
    print(lemma_value + " " + pos + " " + lemma_id)
    for sense in senses:
        print("SENSE: " + str(sense[1]) + "  " + str(check_synset(sense[1])) + "\n")
    out_senses = [s[1] for s in senses]

    hypernyms = hypernyms_word(myword)
    hyponyms = hyponyms_word(myword)
    out_hypernyms = [h[1] for h in hypernyms]
    out_hyponyms = [h[1] for h in hyponyms]
    print("HYPERNYMS: " + str(hypernyms))
    print("n_hypernyms: " + str(len(hypernyms)))
    print("HYPONYMS: " + str(hyponyms))
    print("n_hyponyms: " + str(len(hyponyms)))

    # e.g. "odenet-2339-v" is the synset ID
    # print(str(check_synset("odenet-11159-v")))
    # print(str(check_word_lemma("w2722_4206-v")))
    # print(out_senses)
    # print(out_hyponyms)
    # print(out_hypernyms)
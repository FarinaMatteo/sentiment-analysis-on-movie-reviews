import os
import sys
import nltk
import numpy as np
from sklearn.preprocessing import MinMaxScaler
nltk.download("sentiwordnet")
nltk.download("universal_tagset")
pos2wn = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
from nltk.corpus import sentiwordnet as swn

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.preprocessing import get_movie_reviews_dataset, hconcat

def valence_count(sent):
    """Given a string :param: sent, returns the count of both
    positive and negative tokens in it."""
    tagged_tokens = nltk.pos_tag(sent, tagset="universal")
    tagged_tokens = [(t, pos2wn.get(pos_tag, None)) for (t, pos_tag) in tagged_tokens]
    sentence_counts = {"pos": 0, "neg": 0}
    for (t, pos_tag) in tagged_tokens:
        synsets = swn.senti_synsets(t, pos=pos_tag)
        token_counts = {"pos": 0, "neg": 0}
        for synset in synsets:
            if synset.pos_score() > synset.neg_score():
                token_counts["pos"] += 1
            else:
                token_counts["neg"] += 1
        if token_counts["pos"] > token_counts["neg"]:
            sentence_counts["pos"] += 1
        else:
            sentence_counts["neg"] += 1
    return sentence_counts

def swn_sentence_classification(sent):
    valence_counts = valence_count(sent)
    return 0 if valence_counts["neg"] > valence_counts["pos"] else 1

def diff_pos_neg_feature(doc: str, as_ratio: bool = True) -> list:
    """Returns the DiffPosNeg feature of :param: doc.
    The feature is defined as the numerical distance between sentences
    with a positive orientation and sentences with a negative orientation."""
    pos_count, neg_count = 0, 0
    for sent in doc:
        sent_cls = swn_sentence_classification(sent)
        if sent_cls == 0:
            neg_count += 1
        else:
            pos_count += 1
    if pos_count >= neg_count:
        if as_ratio:
            return [abs(pos_count-neg_count)/(pos_count+neg_count), 1]
        return [abs(pos_count-neg_count), 1]
    if as_ratio:
        return [abs(pos_count-neg_count)/(pos_count+neg_count), 0]
    return [abs(pos_count - neg_count), 0]


class DiffPosNegVectorizer:
    def __init__(self, feature_range=(0,100)) -> None:
        self.min_max_scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, X):
        distances, prevalences = [], []
        for doc in X:
            diffposneg = diff_pos_neg_feature(doc)
            distances.append(diffposneg[0]) # absolute numerical distance
            prevalences.append(diffposneg[1]) # estimated prevalence of neg/pos sentences (replaces the sign!)
        distances = np.expand_dims(np.array(distances), axis=-1)
        distances = self.min_max_scaler.fit_transform(distances)
        prevalences = np.expand_dims(np.array(prevalences), axis=-1)
        return np.int32(hconcat(distances, prevalences))


if __name__ == "__main__":
    neg, pos = get_movie_reviews_dataset()
    corpus = neg + pos
    vectorizer = DiffPosNegVectorizer()
    corpus_diffposneg = vectorizer.fit_transform(corpus)
    print("Neg Corpus DiffPosNeg shape: ", corpus_diffposneg.shape)

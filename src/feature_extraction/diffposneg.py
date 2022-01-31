from utils.preprocessing import get_movie_reviews_dataset, hconcat
from nltk.corpus import sentiwordnet as swn
import os
import sys
import nltk
from nltk.corpus import wordnet
import time
import numpy as np
import multiprocessing as mp
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download("sentiwordnet")
nltk.download("universal_tagset")
pos2wn = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    """Return a synset for an ambiguous word in a context.

    :param iter context_sentence: The context sentence where the ambiguous word
         occurs, passed as an iterable of words.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """

    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        if pos == 'a':
            synsets = [ss for ss in synsets if str(ss.pos()) in ['a', 's']]
        else:
            synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    _, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return sense


def valence_count(sent, tokenizer, memory, update_mem):
    """Given a string :param: sent, returns the count of both
    positive and negative tokens in it."""
    tokens = tokenizer(sent)
    tagged_tokens = nltk.pos_tag(tokens, tagset="universal")
    tagged_tokens = [(t, pos2wn.get(pos_tag, None))
                     for (t, pos_tag) in tagged_tokens]
    sentence_counts = {"pos": 0, "neg": 0}
    for (t, pos_tag) in tagged_tokens:
        token_label = memory.get(t, None)
        if token_label is None:
            token_label = "neg"
            ss = lesk(tokens, t, pos=pos_tag)
            if ss:
                sense = swn.senti_synset(ss.name())
                if sense.pos_score() >= sense.neg_score():
                    token_label = "pos"
            if update_mem:
                memory[t] = token_label
        sentence_counts[token_label] += 1
    return sentence_counts


def swn_sentence_classification(sent, tokenizer, memory, update_mem):
    valence_counts = valence_count(sent, tokenizer, memory, update_mem)
    return 0 if valence_counts["neg"] > valence_counts["pos"] else 1


class DiffPosNegVectorizer(BaseEstimator, TransformerMixin):
    """Class for implementing the DiffPosNeg feature as described in https://aclanthology.org/I13-1114/
    through scikit-learn APIs."""
    
    def __init__(self, tokenizer=word_tokenize, lb=0, ub=1):
        """
        - :param tokenizer: Callable parameter, used to extract tokens from documents
        when vectorizing;
        - :param lb: lower bound for clipping absolute values of numerical distances once scaled;
        - :param rb: same as :param lb:, but upper bound.
        """
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self.tokenizer = tokenizer
        self.lb = lb
        self.ub = ub

    def diff_pos_neg_feature(self, doc, memory, update_mem=False, as_ratio=True) -> list:
        """Returns the DiffPosNeg feature of :param: doc.
        The feature is defined as the numerical distance between sentences
        with a positive orientation and sentences with a negative orientation."""
        pos_count, neg_count = 0, 0
        for sent in sent_tokenize(doc):
            sent_cls = swn_sentence_classification(
                sent, self.tokenizer, memory, update_mem)
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

    def fit(self, X, y=None, **fit_params):
        self.memory_ = {}
        # apply parallel execution of the 'diff_pos_neg' feature extraction function
        with mp.Manager() as manager:
            mem = manager.dict()
            with mp.Pool(processes=mp.cpu_count()) as pool:
                diff_pos_neg_feats = np.array(pool.starmap(
                    self.diff_pos_neg_feature, [(doc, mem, True) for doc in X]))
            self.memory_ = {k: v for k, v in mem.items()}
        distances = diff_pos_neg_feats[:, 0]
        self.min_ = np.amin(distances)
        self.max_ = np.amax(distances)
        return self

    def transform(self, X):
        in_time = time.time()
        # apply parallel execution of the 'diff_pos_neg' feature extraction function
        with mp.Manager() as manager:
            mem = manager.dict()
            mem = {k: v for k, v in self.memory_.items()}
            with mp.Pool(processes=mp.cpu_count()) as pool:
                diff_pos_neg_feats = np.array(pool.starmap(
                    self.diff_pos_neg_feature, [(doc, mem, False) for doc in X]))
        distances = diff_pos_neg_feats[:, 0]
        prevalences = diff_pos_neg_feats[:, -1]

        # scale the values in the range [0,100], taking care of possible values outside the fitted min/max by clipping
        distances = np.clip((distances - self.min_) / (self.max_ -
                            self.min_ + np.finfo(float).eps), a_min=self.lb, a_max=self.ub)
        distances = np.int16(distances*100)

        # put components together and return
        distances = np.expand_dims(distances, axis=-1)
        prevalences = np.expand_dims(np.array(prevalences), axis=-1)
        print(f"Transformed {len(X)} documents in {time.time()-in_time:.2f}s")
        return hconcat(distances, prevalences)

    def fit_transform(self, X, y=None, **fit_params):
        in_time = time.time()
        self.memory_ = {}
        # apply parallel execution of the 'diff_pos_neg' feature extraction function
        with mp.Manager() as manager:
            mem = manager.dict()
            with mp.Pool(processes=mp.cpu_count()) as pool:
                diff_pos_neg_feats = np.array(pool.starmap(
                    self.diff_pos_neg_feature, [(doc, mem, True) for doc in X]))
            self.memory_ = {k: v for k, v in mem.items()}
        distances = diff_pos_neg_feats[:, 0]
        prevalences = diff_pos_neg_feats[:, -1]
        print("Number of positive documents: {}".format(
            np.count_nonzero(prevalences)))

        # override stats inferred from the data
        self.min_ = np.amin(distances)
        self.max_ = np.amax(distances)

        # scaling the values of the distances in the range [0, 1]
        distances = (distances - self.min_) / \
            (self.max_ - self.min_ + np.finfo(float).eps)
        distances = np.int16(distances*100)

        # put the feature components back together after post-processing and return
        distances = np.expand_dims(distances, axis=-1)
        prevalences = np.expand_dims(prevalences, axis=-1)
        print(
            f"Fitted Model and transformed {len(X)} documents in {time.time()-in_time:.2f}s")
        return hconcat(distances, prevalences)


if __name__ == "__main__":
    neg, pos = get_movie_reviews_dataset()
    corpus = neg + pos
    vectorizer = DiffPosNegVectorizer()
    corpus_diffposneg = vectorizer.fit_transform(corpus)
    print("Neg Corpus DiffPosNeg shape: ", corpus_diffposneg.shape)

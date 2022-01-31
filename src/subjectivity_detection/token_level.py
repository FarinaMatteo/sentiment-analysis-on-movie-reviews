from utils.preprocessing import (
    mr2str, get_movie_reviews_dataset, load_corpus_rotten_imdb, vconcat
)
import os
import sys
import math
import time
import nltk
import numpy as np
from joblib import dump
from typing import Callable
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.util import mark_negation
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold

src_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir_path)
repo_root = os.path.dirname(src_dir_path)

UNIVERSAL_TAGSET = ["VERB", "NOUN", "PRON", "ADJ",
                    "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", "."]
SUBJECTIVITY_THRESH = 0.25


def part_of_speech_features(sentence: str, tokenizer: Callable, vocab: set) -> list:
    """Encode the pos tags of the tokens in :param sentence: in a vectorial representation,
    mapping the indices of the tags from the Universal Tagset. :param sentence: is split with the  :param tokenizer:
    callable; resulting tokens not in :param vocab: are filtered out before applying the transformation."""
    tokens = tokenizer(sentence)
    tagged_tokens = nltk.pos_tag(tokens, tagset="universal")
    ret = []
    for tok, tag in tagged_tokens:
        if tok in vocab:
            ret.append(UNIVERSAL_TAGSET.index(tag))
    return ret


def position_features(sentence: str, tokenizer: Callable, vocab: set) -> list:
    """Encodes the relative position of the tokens in :param sentence: as follows:

    - 0 for tokens at the beginning of the sentence; 
    - 1 for tokens in the middle of the sentence;
    - 2 for tokens at the end of the sentence.

    Tokens are extracted with the :param tokenizer: callable and filtered based on
    whether they appear in :param vocab: or not.
    """
    out = []
    tokens = tokenizer(sentence)
    for i, token in enumerate(tokens):
        if token not in vocab:
            continue
        if i == 0:
            out.append(0)
        elif i == len(tokens):
            out.append(2)
        else:
            out.append(1)
    return out


def fl(corpus: list[str], return_vocab: bool = False):
    """Builds the frequency list of a corpus. Returns a dictionary
    where words are the keys and their frequency in the corpus is the respective value.

    If :param return_vocab: is True, the vocabulary of the corpus is returned alongside
    the frequency list."""
    out = {}
    for sent in corpus:
        tokens = word_tokenize(sent)
        for token in tokens:
            if token not in out.keys():
                out[token] = 0
            out[token] += 1
    if return_vocab:
        return out, set(list(out.keys()))
    return out


def dfl(corpus: list[str]) -> dict:
    """Builds the document-based frequency list of a corpus. Returns a dictionary
    where words are the keys and their document-frequency in the corpus is the respective value."""
    out = {}
    for i, sent in enumerate(corpus):
        tokens = word_tokenize(sent)
        for token in tokens:
            if token not in out.keys():
                out[token] = []
            out[token].append(i)
    for token in out.keys():
        out[token] = len(set(out[token]))
    return out


def tf(token, fl):
    return fl[token]


def idf(token, dfl, ndocs):
    return math.log((1+ndocs)/1+dfl[token]) + 1


def tfidf(token, fl, dfl, ndocs):
    return tf(token, fl) * idf(token, dfl, ndocs)


def tfidf_dict(fl, dfl, ndocs, norm=True):
    """Builds a dict with the tfidf (Term-Frequency - Inverse-Document-Frequency) value for each
    word.  

    - :param fl: frequency list of the vocabulary;
    - :param dfl: document-based frequency list of the vocabulary;
    - :ndocs: total number of documents in the corpus;
    - :norm: decide whether to normalize the tfidf values using Euclidean Normalization or not.
    """
    words = set(fl.keys())
    out = {}
    for word in words:
        out[word] = tfidf(word, fl, dfl, ndocs)
    if norm:
        values = normalize(list(out.values()))
        return dict(zip(words, values))
    return out


def normalize(vector):
    """Normalizes :param vector: with Euclidean Normalization."""
    denom = 0
    for item in vector:
        denom += item**2
    denom = math.sqrt(denom)
    for i in range(len(vector)):
        vector[i] /= denom
    return vector


def filter_dict(fl, vocab):
    """Removes entries from the :param fl: frequency list not appearing in :param vocab:."""
    out = {}
    for k, v in fl.items():
        if k in vocab:
            out[k] = v
    return out


def negation_feature(sent: str, tokenizer: Callable, vocab: set) -> list:
    """Encodes :param sent: extracting the Negation Feature. The Negation Feature is
    defined as a vector where a 1 indicates a token being part of a negated phrase and 0 viceversa.

    Tokens are extracted with the :param tokenizer: callable and filtered out based on their appearance
    in :param vocab: before transforming the sentence."""
    tokens = tokenizer(sent)
    valid_tokens = []
    for t in tokens:
        if t in vocab:
            valid_tokens.append(t)
    marked_sent = mark_negation(valid_tokens)
    return [1 if t.endswith("_NEG") else 0 for t in marked_sent]


def token_count(ds, tokenizer, vocab):
    count = 0
    for sent in ds:
        tokens = tokenizer(sent)
        for t in tokens:
            if t in vocab:
                count += 1
    return count


def embed_sentence(sent, tokenizer, vocabulary, tfidf_map):
    """Encodes a sentence extracting  a subset of token-level features 
    w.r.t. the ones proposed in https://arxiv.org/pdf/1312.6962.pdf.

    The features for each token (extracted with the :param tokenizer: callable) of :param sent: are:  
    - its tfidf feature (using :param tfidf_map:);
    - its positional feature;
    - its part_of_speech feature:
    - its negation feature.

    Thus, a matrix of shape (N_tokens, 4) is returned.
    """
    tokens = tokenizer(sent)
    tfidf_feats = []
    position_feats = position_features(sent, tokenizer, vocabulary)
    part_of_speech_feats = part_of_speech_features(sent, tokenizer, vocabulary)
    negation_feats = negation_feature(sent, tokenizer, vocabulary)
    for token in tokens:
        if token in vocabulary:
            tfidf_feats.append(tfidf_map.get(token))

    tfidf_feats = np.expand_dims(np.array(tfidf_feats), axis=-1)
    position_feats = np.expand_dims(np.array(position_feats), axis=-1)
    part_of_speech_feats = np.expand_dims(
        np.array(part_of_speech_feats), axis=-1)
    negation_feats = np.expand_dims(np.array(negation_feats), axis=-1)

    X = np.concatenate((tfidf_feats, part_of_speech_feats,
                       position_feats, negation_feats), axis=1)
    return X


def classify_sentence(clf, sent, subjectivity_thresh, tokenizer, vocabulary, tfidf_map):
    """Performs token-level subjectivity detection on the tokens in :param sent:, then aggregates
    the results for sentence-level classification. If the percentage of subjective tokens exceeds
    :param subjectivity_thresh:, then :param sent: is classified as subjective (objective otherwise)."""
    X = embed_sentence(sent, tokenizer, vocabulary, tfidf_map)
    y = clf.predict(X)
    if np.count_nonzero(y) >= int(len(y)*subjectivity_thresh):
        return 1
    return 0


if __name__ == "__main__":
    in_time = time.time()

    # load the MovieReviews Corpus in order to extract its vocabulary
    neg, pos = get_movie_reviews_dataset(mark_negs=False)
    neg = mr2str(neg)
    pos = mr2str(pos)
    mr = neg + pos
    _, vocabulary = fl(mr, return_vocab=True)

    # load the Rotten IMDB Dataset to train the subjectivity detector
    rotten_imdb_path = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
        'data/rotten_imdb'
    )
    subj, obj = load_corpus_rotten_imdb(rotten_imdb_path)
    dataset = subj + obj

    # define the statistical unigram values for the Rotten IMDB Dataset
    frequency_list = filter_dict(fl(dataset), vocabulary)
    document_fl = filter_dict(dfl(dataset), vocabulary)
    n_docs = len(dataset)
    tfidf_map = tfidf_dict(frequency_list, document_fl, n_docs)

    # build the X matrix embedding the whole Rotten IMDB Dataset
    X = None
    for sent in dataset:
        embedded_sent_matrix = embed_sentence(
            sent, word_tokenize, vocabulary, tfidf_map)
        if X is None:
            X = embedded_sent_matrix
        else:
            X = vconcat(X, embedded_sent_matrix)
    labels = [1]*token_count(subj, word_tokenize, vocabulary) + \
        [0]*token_count(obj, word_tokenize, vocabulary)

    # instantiate and cross-validate the MultinomialNB classifier on the token-level task
    clf = MultinomialNB()
    scores = cross_validate(clf, X, labels,
                            cv=StratifiedKFold(n_splits=10),
                            scoring=['f1_micro'],
                            return_estimator=True,
                            n_jobs=-1)
    average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score: {:.3f}".format(average))

    # grab the best estimator from the cross validation procedure and evaluate it
    # on the sentence-level subjectivity detection task
    estimator = scores['estimator'][np.argmax(
        np.array(scores['test_f1_micro']))]
    y_true = [1]*len(subj) + [0]*len(obj)
    y_pred = [classify_sentence(estimator, sent, SUBJECTIVITY_THRESH,
                                word_tokenize, vocabulary, tfidf_map) for sent in dataset]
    print(classification_report(y_true, y_pred))

    # finally, dump the best estimator on disk
    outpath = os.path.join(repo_root, 'models',
                           'tl_subjectivity_detector.joblib')
    print("Saving model at: ", outpath)
    dump(estimator, outpath)
    out_time = time.time()
    mins = (out_time-in_time)//60
    secs = (out_time-in_time) % 60
    print("Finished in {}m:{}s".format(int(mins), int(secs)))

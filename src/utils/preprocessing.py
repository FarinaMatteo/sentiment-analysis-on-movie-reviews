import nltk
from nltk.sentiment.util import mark_negation
import numpy as np
import os


def lol2str(doc):
    """Transforms a document in the list-of-lists format into
    a block of text (str type)."""
    return " ".join([word for sent in doc for word in sent])

def mr2str(dataset):
    """Transforms the Movie Reviews Dataset (or a slice) into a block of text."""
    return [lol2str(doc) for doc in dataset]

def get_movie_reviews_dataset(mark_negs=True):
    """Uses the nltk library to download the "Movie Reviews" dateset,
    splitting it into negative reviews and positive reviews."""
    nltk.download("movie_reviews")
    from nltk.corpus import movie_reviews
    neg = movie_reviews.paras(categories="neg")
    pos = movie_reviews.paras(categories="pos")
    if mark_negs:
        neg = [[mark_negation(sent) for sent in doc] for doc in neg]
        pos = [[mark_negation(sent) for sent in doc] for doc in pos]
    return neg, pos

def load_corpus_rotten_imdb(path):
    subjective_sentences = "quote.tok.gt9.5000"
    objective_sentences = "plot.tok.gt9.5000"

    subj = []
    with open(os.path.join(path, subjective_sentences), 'r') as f:
        [subj.append(sent.strip()) for sent in f.readlines()]

    obj = []
    with open(os.path.join(path, objective_sentences), 'r') as f:
        [obj.append(sent.strip()) for sent in f.readlines()]

    return subj, obj

def hconcat(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Applies horizontal concatenation to the X1 and X2 matrices, returning the concatenated matrix."""
    assert len(X1.shape) == len(X2.shape) == 2, "function 'hconcat' only works with matrices (np.array with 2 dimensions)."
    assert X1.shape[0] == X2.shape[0], "In order to hconcat matrices, they must have the same number of rows."
    N = X1.shape[0]
    M = X1.shape[1] + X2.shape[1]
    X = np.ndarray(shape=(N,M))
    X[:, :X1.shape[1]] = X1
    X[:, X1.shape[1]:] = X2
    return X

def vconcat(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Applies vertical concatenation to the X1 and X2 matrices, returning the concatenated matrix."""
    assert len(X1.shape) == len(X2.shape) == 2, "function 'vconcat' only works with matrices (np.array with 2 dimensions)."
    assert X1.shape[1] == X2.shape[1], "In order to vconcat matrices, they must have the same number of columns."
    N = X1.shape[0] + X2.shape[0] # sum of 
    M = X1.shape[1]
    X = np.ndarray(shape=(N,M))
    X[:X1.shape[0], :] = X1
    X[X1.shape[0]:, :] = X2
    return X
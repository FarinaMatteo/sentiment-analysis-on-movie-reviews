from pydoc import doc
import nltk
from nltk.sentiment.util import mark_negation
import numpy as np


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

def hconcat(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    N = X1.shape[0]
    M = X1.shape[1] + X2.shape[1]
    X = np.ndarray(shape=(N,M))
    X[:, :X1.shape[1]] = X1
    X[:, X1.shape[1]:] = X2
    return X
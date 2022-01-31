import os
import sys
import time
import sklearn
import logging
import numpy as np
from joblib import dump
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
from feature_extraction.diffposneg import DiffPosNegVectorizer

from utils.preprocessing import hconcat, mr2str

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_basic_logger(logger_name="default"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def switch_vectorizer(vectorizer_name="count"):
    assert vectorizer_name in ("count", "tfidf", "diffposneg", "bert")
    if vectorizer_name == "count":
        return sklearn.feature_extraction.text.CountVectorizer(tokenizer=word_tokenize)
    elif vectorizer_name == "tfidf":
        return sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=word_tokenize)
    elif vectorizer_name == "diffposneg":
        return DiffPosNegVectorizer()


def inference_time(X: list[list[str]], model, vectorizer=None, dim_reducer=None, subj_detector=None, subj_vectorizer=None):
    in_time = time.time()
    if subj_vectorizer and subj_detector:
        subj_features = []
        for i, doc in enumerate(X):
            sents = [" ".join(sent) for sent in doc]
            vectors = subj_vectorizer.transform(sents)
            y_pred = subj_detector.predict(vectors)
            subj_features.append(1 if np.count_nonzero(
                np.array(y_pred)) >= len(y_pred) else 0)
        subj_features = np.array(subj_features)
    if vectorizer:
        X = vectorizer.transform(mr2str(X))
    if isinstance(X, csr_matrix):
        X = X.toarray()
    if dim_reducer:
        X = dim_reducer.transform(X)
    if isinstance(X, csr_matrix):
        X = X.toarray()
    if subj_vectorizer and subj_detector:
        X = hconcat(X, np.expand_dims(subj_features, axis=-1))
    model.predict(X)
    span_time = time.time()-in_time
    return f"{int(span_time//60)}m:{int(span_time%60)}s"


def fit_transform_save(vectorizer, dataset, path):
    X = vectorizer.fit_transform(mr2str(dataset))
    if isinstance(X, csr_matrix):  # in case it is a scipy.csr.csr_matrix
        X = X.toarray()
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    dump(vectorizer, path)
    return vectorizer, X


def join_sents(doc):
    return [[" ".join(word) for word in sent] for sent in doc]

import os
import sys
import time
import sklearn
import logging
from joblib import dump
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extraction.diffposneg import DiffPosNegVectorizer

def get_basic_logger(logger_name="default"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def inference_time(X, model, vectorizer=None):
    in_time = time.time()
    if vectorizer:
        X = vectorizer.transform(X)
    if isinstance(X, csr_matrix):
        X = X.toarray()
    model.predict(X)
    span_time = time.time()-in_time
    return f"{int(span_time//60)}m:{int(span_time%60)}s"
    

def fit_transform_save(vectorizer, dataset, path):
    X = vectorizer.fit_transform(dataset)
    if isinstance(X, csr_matrix): # in case it is a scipy.csr.csr_matrix
        X = X.toarray()
    # once the vectorizer for the 1st stage has been successfully fitted, dump it for further usage
    # NOTE: useful for parallel cross validation!
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    dump(vectorizer, path)
    return vectorizer, X


def join_sents(doc):
    return [[" ".join(word) for word in sent] for sent in doc]
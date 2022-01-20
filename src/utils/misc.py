import os
import sys
import sklearn
import logging
from argparse import ArgumentParser

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
        return sklearn.feature_extraction.text.CountVectorizer()
    elif vectorizer_name == "tfidf":
        return sklearn.feature_extraction.text.TfidfVectorizer()
    elif vectorizer_name == "bert":
        raise NotImplementedError("In the next days...")
    elif vectorizer_name == "diffposneg":
        return DiffPosNegVectorizer()

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--first_stage_vectorizer_name", type=str,
                        help="The name of the vectorizer to be used for the 1st stage classifier (Multinomial NB).\
                            Default: 'count'. Options: 'count', 'tfidf', 'bert'",
                        default="count")
    parser.add_argument("--second_stage_vectorizer_name", type=str,
                        help="The name of the vectorizer to be used for the 2nd stage classifier (Support Vector Classifier).\
                            Default: 'count'. Options: 'count', 'tfidf', 'bert'",
                        default="count")
    parser.add_argument("--kfold_splits", type=int,
                        help="How many folds to use (N-1 for training, 1 for testing) when cross-validating the models.\
                            Default: 5",
                        default=5)
    return parser
    
import os
import sys
import numpy as np
from joblib import dump
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

src_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir_path)
repo_root = os.path.dirname(src_dir_path)
from utils.preprocessing import load_corpus_rotten_imdb, get_movie_reviews_dataset, mr2str


if __name__ == "__main__":

    # load the Rotten IMDB Dataset to train the subjectivity detector
    rotten_imdb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        'data/rotten_imdb'
    )
    subj, obj  = load_corpus_rotten_imdb(rotten_imdb_path)
    dataset = subj + obj

    # load the MovieReviews Corpus in order to extract its vocabulary
    neg, pos = get_movie_reviews_dataset(mark_negs=False)
    neg = mr2str(neg)
    pos = mr2str(pos)
    mr = neg + pos

    vectorizer = TfidfVectorizer()
    vectorizer.fit(mr)
    vectors = vectorizer.transform(dataset)
    labels = [1]*len(subj) + [0]*len(obj)
    clf = MultinomialNB()
    scores = cross_validate(clf, vectors, labels,
                            cv=StratifiedKFold(n_splits=10),
                            scoring=['f1_micro'],
                            return_estimator=True,
                            n_jobs=-1)
    average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score: {:.3f}".format(average))

    assert len(scores["estimator"]) == len(scores["test_f1_micro"])
    estimator = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    y_pred = estimator.predict(vectors)
    print(classification_report(labels, y_pred))

    # finally, dump the best estimator on disk
    outpath = os.path.join(repo_root, 'models', 'sl_subjectivity_detector.joblib')
    print("Saving model at: ", outpath)
    dump(estimator, outpath)
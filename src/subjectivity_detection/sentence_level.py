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


def build_argparser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-r", "--representation", type=str,
                        default="count",
                        help="The Document representation to train the subjectivity detector. Choose between: 'count' or 'tfidf'. Default 'count'.")
    parser.add_argument("-clf", "--classifier", type=str, default="multinomial",
                        help="The Naive Bayes classifier to train. Choose between 'multinomial' or 'bernoulli'. Default: 'multinomial'.")
    return parser


def main(representation="count", classifier="multinomial"):
    assert representation in ("count", "tfidf")
    assert classifier in ("multinomial", "bernoulli")

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

    # instantiate vectorizer and classifier based on passed arguments
    if representation == "count":
        vectorizer = CountVectorizer()
    else:
        vectorizer = TfidfVectorizer()

    if classifier == "multinomial":
        clf = MultinomialNB()
    else:
        clf = BernoulliNB()

    # fit the vectorizer on the MovieReviews dataset to link it to its vocab
    # MovieReviews will be the target dataset for the final evaluation!
    vectorizer.fit(mr)
    
    # then vectorize the RottenIMDB Dataset with the vocab constraints from MovieReviews 
    vectors = vectorizer.transform(dataset)
    labels = [1]*len(subj) + [0]*len(obj)

    # perform cross validation and grab the best estimator
    scores = cross_validate(clf, vectors, labels,
                            cv=StratifiedKFold(n_splits=10),
                            scoring=['f1_micro'],
                            return_estimator=True,
                            n_jobs=-1)
    estimator = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    
    # display cross validation empirical results on the F1 metric
    average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("Average F1 Score from cross validation: {:.2f}".format(average))
    
    # test the best estimator on the whole RottenIMBD dataset and display the output
    y_pred = estimator.predict(vectors)
    print(classification_report(labels, y_pred))

    # finally, dump the best estimator on disk
    outpath = os.path.join(repo_root, 'models',
                            f'{representation}_{classifier}_subj_det.joblib')
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    print("Saving model at: ", outpath)
    dump(estimator, outpath)

if __name__ == "__main__":
    parser = build_argparser()
    main(**vars(parser.parse_args()))
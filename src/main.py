import numpy as np
from model.two_stage_classifier import TwoStageClassifier
from utils.misc import switch_vectorizer, build_argparser
from utils.preprocessing import get_movie_reviews_dataset, mr2str, hconcat
from sklearn.model_selection import cross_validate, StratifiedKFold


def main(first_stage_vectorizer_name="count", second_stage_vectorizer_name="tfidf", kfold_splits=5):
    # download and preprocess the MovieReviews dataset
    neg, pos = get_movie_reviews_dataset()
    if first_stage_vectorizer_name != "diffposneg":
        neg_corpus = mr2str(neg)
        pos_corpus = mr2str(pos)
    else:
        neg_corpus = neg
        pos_corpus = pos
    
    mr_corpus = neg_corpus + pos_corpus
    # represent documents for the first stage classifier (NB) with the selected 
    # vectorization method
    first_stage_vectorizer = switch_vectorizer(first_stage_vectorizer_name)
    X_first_stage = first_stage_vectorizer.fit_transform(mr_corpus)
    if not isinstance(X_first_stage, np.ndarray): # in case it is a scipy.csr.csr_matrix
        X_first_stage = X_first_stage.toarray()
    print("X 1st stage shape: ", X_first_stage.shape)

    # define the labels of the MovieReviews dataset
    y = np.array([0]*len(neg_corpus) + [1]*len(pos_corpus))
    print("Labels shape: ", y.shape)

    # instantiate and test the Two Stage Classifier
    if first_stage_vectorizer_name != second_stage_vectorizer_name:
        # in case the vectorizers for the 1st and 2nd stages are different, we
        # must let the classifier know how to split the merged dataset
        clf = TwoStageClassifier(second_stage_starting_point=X_first_stage.shape[1])

        # vectorize also the 2nd stage dataset
        if first_stage_vectorizer_name == "diffposneg":
            neg_corpus = mr2str(neg_corpus)
            pos_corpus = mr2str(pos_corpus)
            mr_corpus = neg_corpus + pos_corpus
        second_stage_vectorizer = switch_vectorizer(second_stage_vectorizer_name)
        X_second_stage = second_stage_vectorizer.fit_transform(mr_corpus)
        if not isinstance(X_second_stage, np.ndarray):
            X_second_stage = X_second_stage.toarray()

        # concatenate the different matrices
        X_second_stage = hconcat(X_first_stage, X_second_stage)
        X = hconcat(X_first_stage, X_second_stage)
    else:
        # in case the same vectorizer was selected for both the 1st and 2nd stages,
        # there is no need to vectorize once more
        clf = TwoStageClassifier()
        X_second_stage = X_first_stage.copy()
        X = X_first_stage.copy()
    
    print("X 2nd stage shape: ", X_second_stage.shape)
    print("X shape: ", X.shape)

    # evaluate the two-stage-classifier on the merged dataset
    # NOTE: the dataset is 'merged' only if different vectorizers were selected
    scores = cross_validate(clf, X, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'])
    average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the Two Stage Classifier: {:.3f}".format(average))

    # test the Multinomial Naive Bayes only
    scores = cross_validate(clf.first_stage_clf, X_first_stage, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'])
    average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the Multinomial Naive Bayes: {:.3f}".format(average))

    # test the SVC only
    scores = cross_validate(clf.second_stage_clf, X_second_stage, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'])
    average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the SVC: {:.3f}".format(average))


if __name__ == "__main__":
    parser = build_argparser()
    main(**vars(parser.parse_args()))
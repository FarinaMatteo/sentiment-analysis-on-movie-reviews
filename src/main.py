import os
import time
import numpy as np
from tqdm import tqdm
from joblib import load
from copy import deepcopy
from itertools import compress
from utils.misc import switch_vectorizer, inference_time, fit_transform_save
from model.two_stage_classifier import TwoStageClassifier
from utils.preprocessing import get_movie_reviews_dataset, mr2str
from sklearn.model_selection import cross_validate, StratifiedKFold

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(repo_root, "models")


def main(first_stage_vectorizer_name="count",
        second_stage_vectorizer_name="tfidf",
        subj_det=True,
        subj_det_filename="count_bernoulli",
        kfold_splits=5):
    
    in_time = time.time()

    # download and preprocess the MovieReviews dataset
    neg, pos = get_movie_reviews_dataset(mark_negs=False)
    mr = list(neg + pos)
    neg_corpus = mr2str(neg)
    pos_corpus = mr2str(pos)
    mr_corpus = neg_corpus + pos_corpus

    # define the labels of the MovieReviews dataset
    y = np.array([0]*len(neg_corpus) + [1]*len(pos_corpus))
    print("Labels shape: ", y.shape)

    # apply sentence level subjectivity detection to reject objective sentences within documents
    if subj_det:
        assert len(subj_det_filename.split("_")) == 2
        assert subj_det_filename.split("_")[0] in ("count", "tfidf")
        assert subj_det_filename.split("_")[1] in ("bernoulli", "multinomial")
        subj_vectorizer = load(os.path.join(models_dir, f"{subj_det_filename}_subj_det_vectorizer.joblib"))
        subj_detector = load(os.path.join(models_dir, f"{subj_det_filename}_subj_det_model.joblib"))

        # classify each sentence of each document in the MovieReviews dataset. Then, prune
        # the factual ones.
        n_removed, total = 0, 0
        for i, doc in enumerate(deepcopy(mr)):
            sents = [" ".join(sent) for sent in doc]
            vectors = subj_vectorizer.transform(sents)
            y_pred = subj_detector.predict(vectors)
            mr[i] = list(compress(doc, y_pred))
            n_removed += len(doc) - len(mr[i])
            total += len(doc)
        print(f"Removed {n_removed} sents out of {total} thanks to subjectivity detection.")
        mr_corpus = mr2str(mr)
    
    # represent documents for the first stage classifier (NB) with the selected 
    # vectorization method
    first_stage_vectorizer = switch_vectorizer(first_stage_vectorizer_name)
    first_stage_vectorizer_path = os.path.join(models_dir, "first_stage_vectorizer.joblib")
    first_stage_vectorizer, X = fit_transform_save(first_stage_vectorizer, mr_corpus, first_stage_vectorizer_path)

    # repeat the exact same operations for the 2nd vectorizer
    second_stage_vectorizer = switch_vectorizer(second_stage_vectorizer_name)
    second_stage_vectorizer_path = os.path.join(models_dir, "second_stage_vectorizer.joblib")
    second_stage_vectorizer, X_second_stage = fit_transform_save(second_stage_vectorizer, mr_corpus, second_stage_vectorizer_path)

    # instantiate the custom Two Stage Classifier with the pre-trained vectorizers
    clf = TwoStageClassifier(
        first_stage_vectorizer_path=first_stage_vectorizer_path,
        second_stage_vectorizer_path=second_stage_vectorizer_path,
    )

    # evaluate the Two Stage Classifier
    scores = cross_validate(clf, mr_corpus, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'],
                            return_estimator=True)
    two_stage_clf_average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the Two Stage Classifier: {:.3f}".format(two_stage_clf_average))
    
    # test the inference time for the Two Stage classifier
    best = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    two_stage_clf_exec_time = inference_time(mr_corpus, best)

    # test the Multinomial Naive Bayes only
    scores = cross_validate(clf.first_stage_clf, X, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'],
                            n_jobs=-1,
                            return_estimator=True)
    multinomial_nb_average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the Multinomial Naive Bayes: {:.3f}".format(multinomial_nb_average))
    
    # test the inference time for the Naive Bayes classifier
    best = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    multinomial_nb_exec_time = inference_time(mr_corpus, best, first_stage_vectorizer)

    # test the SVC only
    scores = cross_validate(clf.second_stage_clf, X_second_stage, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'],
                            return_estimator=True,
                            n_jobs=-1)
    svc_average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the SVC: {:.3f}".format(svc_average))
    
    # test the inference time for the SVC
    best = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    svc_exec_time = inference_time(mr_corpus, best, second_stage_vectorizer)

    # print the total execution time on the terminal
    out_time = time.time()
    mins = int((out_time - in_time) // 60)
    secs = int((out_time - in_time) % 60)
    print(f"Finished in {mins}m:{secs}s.")

    # return performances and metrics in case of an evaluation
    return [
        first_stage_vectorizer_name,
        second_stage_vectorizer_name,
        subj_det,
        round(two_stage_clf_average, 3),
        round(multinomial_nb_average, 3),
        round(svc_average, 3),
        two_stage_clf_exec_time,
        multinomial_nb_exec_time,
        svc_exec_time,
    ]


if __name__ == "__main__":
    import pandas as pd

    # define possible options for the experiment parameters
    first_stage_vec_options = ["diffposneg", "count"]
    second_stage_vec_options = [["count", "tfidf"], ["tfidf"]]
    subj_det_options = [True, False]

    total = sum([len(opts)*len(subj_det_options) for opts in second_stage_vec_options])
    with tqdm(total=total) as pbar:
        # exhaustive search an all possible combinations
        summary = []
        for i, first_stage_vec_option in enumerate(first_stage_vec_options):
            for second_stage_vec_option in second_stage_vec_options[i]:
                for subj_det_option in subj_det_options:
                    params = {
                        "first_stage_vectorizer_name": first_stage_vec_option,
                        "second_stage_vectorizer_name": second_stage_vec_option,
                        "subj_det": subj_det_option,
                    }
                    data = main(**params)
                    summary.append(data)
                    pbar.update()

    # finally, save the results of the experiments on disk
    summary_path = os.path.join(repo_root, "csv-data", "summary.csv")
    if not os.path.exists(os.path.dirname(summary_path)):
        os.makedirs(os.path.dirname(summary_path))
    print("Saving summary at: ", summary_path)
    columns = ["vec1", "vec2", "subjDet", "2Stage F1", "Multinomial F1", "SVC F1", "2Stage InfTime", "Multinomial Inftime", "SVC Inftime"]
    pd.DataFrame(summary, columns=columns).to_csv(summary_path, index=False)
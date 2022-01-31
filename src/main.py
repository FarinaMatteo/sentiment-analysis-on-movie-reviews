import os
import time
import numpy as np
from joblib import load
from copy import deepcopy
from itertools import compress
from model.two_stage_classifier import TwoStageClassifier
from utils.preprocessing import get_movie_reviews_dataset, hconcat
from sklearn.model_selection import cross_validate, StratifiedKFold
from utils.misc import switch_vectorizer, inference_time, fit_transform_save

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(repo_root, "models")
dim_reducer_path = os.path.join(models_dir, "dim_reducer.joblib")


def main(first_stage_vectorizer_name="count",
         second_stage_vectorizer_name="tfidf",
         subj_det="filter",
         subj_det_filename="count_bernoulli",
         dim_red=True,
         kfold_splits=5):

    in_time = time.time()

    assert subj_det in ("filter", "aggregate")
    assert len(subj_det_filename.split("_")) == 2
    assert subj_det_filename.split("_")[0] in ("count", "tfidf")
    assert subj_det_filename.split("_")[1] in ("bernoulli", "multinomial")

    # download and preprocess the MovieReviews dataset
    neg, pos = get_movie_reviews_dataset(mark_negs=False)
    mr = list(neg + pos)

    # define the labels of the MovieReviews dataset
    y = np.array([0]*len(neg) + [1]*len(pos))
    print("Labels shape: ", y.shape)

    # apply sentence level subjectivity detection to reject objective sentences within documents
    subj_vectorizer_path = os.path.join(
        models_dir, f"{subj_det_filename}_subj_det_vectorizer.joblib")
    subj_vectorizer = load(subj_vectorizer_path)
    subj_detector_path = os.path.join(
        models_dir, f"{subj_det_filename}_subj_det_model.joblib")
    subj_detector = load(subj_detector_path)
    subj_filter_mins, subj_filter_secs = 0, 0
    if subj_det == "filter":
        # classify each sentence of each document in the MovieReviews dataset. Then, prune
        # the factual ones.
        subj_filter_in_time = time.time()
        n_removed, total = 0, 0
        for i, doc in enumerate(deepcopy(mr)):
            sents = [" ".join(sent) for sent in doc]
            vectors = subj_vectorizer.transform(sents)
            y_pred = subj_detector.predict(vectors)
            mr[i] = list(compress(doc, y_pred))
            n_removed += len(doc) - len(mr[i])
            total += len(doc)
        print(
            f"Removed {n_removed} sents out of {total} thanks to subjectivity detection.")
        subj_filter_span_time = time.time() - subj_filter_in_time
        subj_filter_mins = int(subj_filter_span_time//60)
        subj_filter_secs = int(subj_filter_span_time % 60)

    # represent documents for the first stage classifier (NB) with the selected
    # vectorization method
    first_stage_vectorizer = switch_vectorizer(first_stage_vectorizer_name)
    first_stage_vectorizer_path = os.path.join(
        models_dir, "first_stage_vectorizer.joblib")
    first_stage_vectorizer, X = fit_transform_save(
        first_stage_vectorizer, mr, first_stage_vectorizer_path)

    # repeat the exact same operations for the 2nd vectorizer
    second_stage_vectorizer = switch_vectorizer(second_stage_vectorizer_name)
    second_stage_vectorizer_path = os.path.join(
        models_dir, "second_stage_vectorizer.joblib")
    second_stage_vectorizer, X_second_stage = fit_transform_save(
        second_stage_vectorizer, mr, second_stage_vectorizer_path)

    # instantiate the custom Two Stage Classifier with the pre-trained vectorizers
    two_stage_clf_params = {
        "first_stage_vectorizer_path": first_stage_vectorizer_path,
        "second_stage_vectorizer_path": second_stage_vectorizer_path,
        "dim_red": dim_red
    }
    if subj_det == "aggregate":
        two_stage_clf_params["use_subjectivity"] = True
        two_stage_clf_params["subj_vectorizer_path"] = subj_vectorizer_path
        two_stage_clf_params["subj_detector_path"] = subj_detector_path

    clf = TwoStageClassifier(**two_stage_clf_params)

    # evaluate the Two Stage Classifier
    scores = cross_validate(clf, mr, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'],
                            return_estimator=True)
    two_stage_clf_average = sum(
        scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the Two Stage Classifier: {:.3f}".format(
        two_stage_clf_average))

    # test the inference time for the Two Stage classifier
    best_two_stage = scores["estimator"][np.argmax(
        np.array(scores["test_f1_micro"]))]
    two_stage_clf_exec_time = inference_time(mr, best_two_stage)

    # test the Multinomial Naive Bayes only
    scores = cross_validate(clf.first_stage_clf, X, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'],
                            n_jobs=-1,
                            return_estimator=True)
    multinomial_nb_average = sum(
        scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the Multinomial Naive Bayes: {:.3f}".format(
        multinomial_nb_average))

    # test the inference time for the Naive Bayes classifier
    best = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    multinomial_nb_exec_time = inference_time(mr, best, first_stage_vectorizer)
    if subj_det == "filter":
        multinomial_nb_mins = int(multinomial_nb_exec_time.split(":")[
                                  0][:-1]) + subj_filter_mins
        multinomial_nb_secs = int(multinomial_nb_exec_time.split(":")[
                                  1][:-1]) + subj_filter_secs
        multinomial_nb_exec_time = f"{multinomial_nb_mins}m:{multinomial_nb_secs}s"

    # test the SVC only
    if dim_red:
        X_second_stage = best_two_stage.dim_reducer.transform(X_second_stage)

    if subj_det == "aggregate":
        subj_features = []
        for i, doc in enumerate(deepcopy(mr)):
            sents = [" ".join(sent) for sent in doc]
            vectors = best_two_stage.subj_vectorizer.transform(sents)
            y_pred = best_two_stage.subj_detector.predict(vectors)
            subj_features.append(1 if np.count_nonzero(
                np.array(y_pred)) >= len(y_pred) else 0)
        subj_features = np.array(subj_features)
        X_second_stage = hconcat(
            X_second_stage, np.expand_dims(subj_features, axis=-1))

    scores = cross_validate(clf.second_stage_clf, X_second_stage, y,
                            cv=StratifiedKFold(n_splits=kfold_splits),
                            scoring=['f1_micro'],
                            return_estimator=True,
                            n_jobs=-1)
    svc_average = sum(scores['test_f1_micro'])/len(scores['test_f1_micro'])
    print("F1 Score for the SVC: {:.3f}".format(svc_average))

    # test the inference time for the SVC
    best = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    svc_exec_time = inference_time(mr, best, second_stage_vectorizer,
                                   dim_reducer=best_two_stage.dim_reducer if dim_red else None,
                                   subj_detector=best_two_stage.subj_detector if subj_det == "aggregate" else None,
                                   subj_vectorizer=best_two_stage.subj_vectorizer if subj_det == "aggregate" else None)

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
        dim_red,
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
    subj_det_options = ["aggregate", "filter"]
    dim_red_options = [True, False]

    # exhaustive search an all possible combinations
    summary = []
    for i, first_stage_vec_option in enumerate(first_stage_vec_options):
        for second_stage_vec_option in second_stage_vec_options[i]:
            for subj_det_option in subj_det_options:
                for dim_red_option in dim_red_options:
                    params = {
                        "first_stage_vectorizer_name": first_stage_vec_option,
                        "second_stage_vectorizer_name": second_stage_vec_option,
                        "subj_det": subj_det_option,
                        "dim_red": dim_red_option
                    }
                    data = main(**params)
                    summary.append(data)

    # finally, save the results of the experiments on disk
    summary_path = os.path.join(repo_root, "csv-data", "test.csv")
    if not os.path.exists(os.path.dirname(summary_path)):
        os.makedirs(os.path.dirname(summary_path))
    print("Saving summary at: ", summary_path)
    columns = ["vec1", "vec2", "subjDet", "dimRed", "2Stage F1", "Multinomial F1",
               "SVC F1", "2Stage InfTime", "Multinomial Inftime", "SVC Inftime"]
    pd.DataFrame(summary, columns=columns).to_csv(summary_path, index=False)

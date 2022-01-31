import os
import time
import numpy as np
from joblib import load
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from utils.misc import get_basic_logger
from sklearn.naive_bayes import MultinomialNB
from utils.preprocessing import hconcat, mr2str
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger = get_basic_logger("TwoStageClassifier")
repo_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

models_dir = os.path.join(repo_root, "models")

first_stage_vectorizer_path = os.path.join(
    models_dir, "first_stage_vectorizer.joblib")

second_stage_vectorizer_path = os.path.join(
    models_dir, "second_stage_vectorizer.joblib")

dim_reducer_path = os.path.join(models_dir, "dim_reducer.joblib")

subj_detector_path = os.path.join(
    models_dir, "count_bernoulli_subj_det_model.joblib")

subj_vectorizer_path = os.path.join(
    models_dir, "count_bernoulli_subj_det_vectorizer.joblib")


class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, first_stage_vectorizer_path=first_stage_vectorizer_path,
                 second_stage_vectorizer_path=second_stage_vectorizer_path,
                 neg_cls_min_confidence=0.6,
                 pos_cls_min_confidence=0.6,
                 dim_red=True,
                 use_subjectivity=False,
                 subj_detector_path=subj_detector_path,
                 subj_vectorizer_path=subj_vectorizer_path):

        assert 0 < neg_cls_min_confidence < 1, \
            "The min confidence for the Negative Class must be within (0,1)"
        assert 0 < pos_cls_min_confidence < 1, \
            "The min confidence for the Positive Class must be wihtin (0,1)"
        self.first_stage_clf = MultinomialNB()
        self.second_stage_clf = SVC()
        self.neg_cls_min_confidence = neg_cls_min_confidence
        self.pos_cls_min_confidence = pos_cls_min_confidence
        self.first_stage_vectorizer_path = first_stage_vectorizer_path
        self.second_stage_vectorizer_path = second_stage_vectorizer_path
        self.first_stage_vectorizer = load(self.first_stage_vectorizer_path)
        self.second_stage_vectorizer = load(self.second_stage_vectorizer_path)
        self.dim_red = dim_red
        self.dim_reducer = None
        if self.dim_red:
            self.dim_reducer = LinearDiscriminantAnalysis()
        self.use_subjectivity = use_subjectivity
        self.subj_detector_path = subj_detector_path
        self.subj_vectorizer_path = subj_vectorizer_path
        if self.use_subjectivity:
            self.subj_detector = load(self.subj_detector_path)
            self.subj_vectorizer = load(self.subj_vectorizer_path)

    def subjectivity_features(self, tokenized_corpus):
        feats = []
        for i, doc in enumerate(tokenized_corpus):
            sents = [" ".join(sent) for sent in doc]
            vectors = self.subj_vectorizer.transform(sents)
            y_pred = self.subj_detector.predict(vectors)
            feats.append(1 if np.count_nonzero(
                np.array(y_pred)) >= len(y_pred) else 0)
        return np.array(feats)

    def fit_first_stage_vectorizer(self, X, transform=False):
        if transform:
            return self.first_stage_vectorizer.fit_transform(X)
        return self.first_stage_vectorizer.fit(X)

    def fit_second_stage_vectorizer(self, X, transform=False):
        if transform:
            return self.second_stage_vectorizer.fit_transform(X)
        return self.second_stage_vectorizer.fit(X)

    def fit_first_stage(self, X: np.ndarray, y: np.ndarray) -> None:
        check_is_fitted(self.first_stage_vectorizer,
                        msg="""Make sure to fit the vectorizer of the 1st stage before fitting the respective classifier.
                        You can do it by calling the 'fit_first_stage_vectorizer' method.""")
        logger.info("Fitting 1st Stage Classifier")
        self.first_stage_clf.fit(X, y)

    def fit_second_stage(self, X: np.ndarray, y: np.ndarray, tokenized_corpus=None) -> None:
        check_is_fitted(self.second_stage_vectorizer,
                        msg="""Make sure to fit the vectorizer of the 2nd stage before fitting the respective classifier.
                        You can do it by calling the 'fit_second_stage_vectorizer' method.""")
        if isinstance(X, csr_matrix):
            X = X.toarray()
        if self.dim_red:
            logger.info("Fitting dimensionality reducer")
            X = self.dim_reducer.fit_transform(X, y)

        if self.use_subjectivity and tokenized_corpus is not None:
            subj_features = self.subjectivity_features(tokenized_corpus)
            X = hconcat(X, np.expand_dims(subj_features, -1))

        logger.info("Fitting 2nd Stage Classifier")
        self.second_stage_clf.fit(X, y)

    def fit(self, X: list[list[str]], y: np.ndarray) -> None:
        in_time = time.time()
        X_los = mr2str(X)

        X_first_stage = self.first_stage_vectorizer.transform(X_los)
        self.fit_first_stage(X_first_stage, y)

        X_second_stage = self.second_stage_vectorizer.transform(X_los)
        self.fit_second_stage(X_second_stage, y, X)

        logger.info("Fitting finished successfully. Total time={:.2f}s".format(
            time.time()-in_time))
        return self

    def reject_option(self, y) -> bool:
        cls = np.argmax(y)
        if cls == 0 and y[cls] < self.neg_cls_min_confidence:  # negative case
            return True
        if cls == 1 and y[cls] < self.pos_cls_min_confidence:  # positive case
            return True
        return False

    def predict(self, X: list[list[str]]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        Y = []
        X_los = mr2str(X)
        X_transformed = self.first_stage_vectorizer.transform(X_los)
        C_first_stage = self.first_stage_clf.predict_proba(X_transformed)
        rejected_samples_los, rejected_samples_lol, rejected_indices = [], [], []
        for i, cls_proba in enumerate(C_first_stage):
            if self.reject_option(cls_proba) and self.second_stage_vectorizer is not None:
                rejected_sample_los = X_los[i]
                rejected_samples_los.append(
                    rejected_sample_los)  # i-th document in X
                rejected_sample_lol = X[i]
                rejected_samples_lol.append(rejected_sample_lol)
                rejected_indices.append(i)
            else:
                y = np.argmax(cls_proba)
                Y.append(y)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples_los) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(
                rejected_samples_los)
            if isinstance(rejected_vectors, csr_matrix):
                rejected_vectors = rejected_vectors.toarray()
            if self.dim_red:
                rejected_vectors = self.dim_reducer.transform(rejected_vectors)
            if self.use_subjectivity:
                subj_features = self.subjectivity_features(
                    rejected_samples_lol)
                rejected_vectors = hconcat(
                    rejected_vectors, np.expand_dims(subj_features, axis=-1))
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                Y.insert(rejected_idx, y)

        logger.info(
            f"SVC was called {len(rejected_samples_los)/len(X)*100:.1f}% of the times.")

        # memory cleanup and return
        del C_first_stage
        del rejected_samples_los
        del rejected_indices
        return np.array(Y)

    def predict_proba(self, X: list[list[str]]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_los = mr2str(X)
        X_transformed = self.first_stage_vectorizer.transform(X_los)
        C = self.first_stage_clf.predict_proba(X_transformed)
        rejected_samples_los, rejected_samples_lol, rejected_indices = [], [], []
        for i, c in enumerate(C):
            if self.reject_option(c) and self.second_stage_vectorizer is not None:
                rejected_sample_los = X_los[i]
                rejected_samples_los.append(
                    rejected_sample_los)  # i-th document in X
                rejected_sample_lol = X[i]
                rejected_samples_lol.append(rejected_sample_lol)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples_los) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(
                rejected_samples_los)
            if isinstance(rejected_vectors, csr_matrix):
                rejected_vectors = rejected_vectors.toarray()
            if self.dim_red:
                rejected_vectors = self.dim_reducer.transform(rejected_vectors)
            if self.use_subjectivity:
                subj_features = self.subjectivity_features(
                    rejected_samples_lol)
                rejected_vectors = hconcat(
                    rejected_vectors, np.expand_dims(subj_features, axis=-1))
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a class probability of 1 to the class predicted by the SVM
                # 0 probability otherwise
                cls_probs = np.array([.0, .0])
                cls = np.argmax(y)
                cls_probs[cls] = 1.
                C[rejected_idx] = cls_probs

        logger.info(
            f"SVC was called {len(rejected_samples_los)/len(X)*100:.1f}% of the times.")

        # memory cleanup and return
        del rejected_samples_los
        del rejected_indices
        return np.array(C)

    def predict_log_proba(self, X: list[list[str]]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_los = mr2str(X)
        X_transformed = self.first_stage_vectorizer.transform(X_los)
        C = self.first_stage_clf.predict_log_proba(X_transformed)
        rejected_samples_los, rejected_samples_lol, rejected_indices = [], [], []
        for i, c in enumerate(C):
            if self.reject_option(c) and self.second_stage_vectorizer is not None:
                rejected_sample_los = X_los[i]
                rejected_samples_los.append(
                    rejected_sample_los)  # i-th document in X
                rejected_sample_lol = X[i]
                rejected_samples_lol.append(rejected_sample_lol)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples_los) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(
                rejected_samples_los)
            if isinstance(rejected_vectors, csr_matrix):
                rejected_vectors = rejected_vectors.toarray()
            if self.dim_red:
                rejected_vectors = self.dim_reducer.transform(rejected_vectors)
            if self.use_subjectivity:
                subj_features = self.subjectivity_features(
                    rejected_samples_lol)
                rejected_vectors = hconcat(
                    rejected_vectors, np.expand_dims(subj_features, axis=-1))
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a log-prob of 0 to the predicted class (log(1))
                # assign negative infinity to the other class (approx. log(0))
                log_probs = np.array([-np.inf, -np.inf])
                cls = np.argmax(y)
                log_probs[cls] = 0
                C[rejected_idx] = log_probs

        logger.info(
            f"SVC was called {len(rejected_samples_los)/len(X)*100:.1f}% of the times.")

        # memory cleanup and return
        del rejected_samples_los
        del rejected_indices
        return np.array(C)

    def score(self, X: list[list[str]], y):
        y_pred = self.predict(X)
        binary_acc_vector = [1 if y_gt_ == y_pred_ else 0 for (
            y_gt_, y_pred_) in zip(y, y_pred)]
        return sum(binary_acc_vector) / len(binary_acc_vector)

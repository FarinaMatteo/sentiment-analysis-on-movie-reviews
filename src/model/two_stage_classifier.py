import os
import time
import numpy as np
from joblib import load
from utils.misc import get_basic_logger
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin

logger = get_basic_logger("TwoStageClassifier")
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_dir = os.path.join(repo_root, "models")
first_stage_vectorizer_path = os.path.join(models_dir, "first_stage_vectorizer.joblib")
second_stage_vectorizer_path = os.path.join(models_dir, "second_stage_vectorizer.joblib")

class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, first_stage_vectorizer_path=first_stage_vectorizer_path, second_stage_vectorizer_path=second_stage_vectorizer_path,
                    neg_cls_min_confidence=0.6, pos_cls_min_confidence=0.6, dim_red=False):
        assert 0 < neg_cls_min_confidence < 1, "The min confidence for the Negative Class must be within (0,1)"
        assert 0 < pos_cls_min_confidence < 1, "The min confidence for the Positive Class must be wihtin (0,1)"
        self.first_stage_clf = MultinomialNB()
        self.second_stage_clf = LinearSVC()
        self.neg_cls_min_confidence = neg_cls_min_confidence
        self.pos_cls_min_confidence = pos_cls_min_confidence
        self.first_stage_vectorizer_path = first_stage_vectorizer_path
        self.second_stage_vectorizer_path = second_stage_vectorizer_path
        self.first_stage_vectorizer = load(self.first_stage_vectorizer_path)
        self.second_stage_vectorizer = load(self.second_stage_vectorizer_path)


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

    def fit_second_stage(self, X: np.ndarray, y: np.ndarray) -> None:
        check_is_fitted(self.second_stage_vectorizer, 
                        msg="""Make sure to fit the vectorizer of the 2nd stage before fitting the respective classifier.
                        You can do it by calling the 'fit_second_stage_vectorizer' method.""")
        logger.info("Fitting 2nd Stage Classifier")
        self.second_stage_clf.fit(X, y)

    def fit(self, X: list[str], y: np.ndarray) -> None:
        in_time = time.time()
        
        X_first_stage = self.first_stage_vectorizer.transform(X)
        self.fit_first_stage(X_first_stage, y)
        
        X_second_stage = self.second_stage_vectorizer.transform(X)
        self.fit_second_stage(X_second_stage, y)
        
        logger.info("Fitting finished successfully. Total time={:.2f}s".format(time.time()-in_time))
        return self

    def reject_option(self, y) -> bool:
        cls = np.argmax(y)
        if cls == 0 and y[cls] < self.neg_cls_min_confidence: # negative case
            return True
        if cls == 1 and y[cls] < self.pos_cls_min_confidence: # positive case
            return True
        return False

    def predict(self, X: list[str]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        Y = []
        X_transformed = self.first_stage_vectorizer.transform(X)
        C_first_stage = self.first_stage_clf.predict_proba(X_transformed)
        rejected_samples, rejected_indices = [], []
        for i, cls_proba in enumerate(C_first_stage):
            if self.reject_option(cls_proba) and self.second_stage_vectorizer is not None:
                rejected_sample = X[i]
                rejected_samples.append(rejected_sample) # i-th document in X
                rejected_indices.append(i)
            else:
                y = np.argmax(cls_proba)
                Y.append(y)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(rejected_samples)
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                Y.insert(rejected_idx, y)

        logger.info(f"LinearSVC was called {len(rejected_samples)/len(X)*100:.1f}% of the times.")
        
        # memory cleanup and return
        del C_first_stage
        del rejected_samples
        del rejected_indices
        return np.array(Y)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_transformed = self.first_stage_vectorizer.transform(X)
        C = self.first_stage_clf.predict_proba(X_transformed)
        rejected_samples, rejected_indices = [], []
        for i, c in enumerate(C):
            if self.reject_option(c) and self.second_stage_vectorizer is not None:
                rejected_sample = X[i]
                rejected_samples.append(rejected_sample)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(rejected_samples)
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a class probability of 1 to the class predicted by the SVM
                # 0 probability otherwise
                cls_probs = np.array([.0,.0])
                cls = np.argmax(y)
                cls_probs[cls] = 1.
                C[rejected_idx] = cls_probs

        logger.info(f"LinearSVC was called {len(rejected_samples)/len(X)*100:.1f}% of the times.")

        # memory cleanup and return
        del rejected_samples
        del rejected_indices
        return np.array(C)

    def predict_log_proba(self, X) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_transformed = self.first_stage_vectorizer.transform(X)
        C = self.first_stage_clf.predict_log_proba(X_transformed)
        rejected_samples, rejected_indices = [], []
        for i, c in enumerate(C):
            if self.reject_option(c) and self.second_stage_vectorizer is not None:
                rejected_sample = X[i]
                rejected_samples.append(rejected_sample)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(rejected_samples)
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a log-prob of 0 to the predicted class (log(1))
                # assign negative infinity to the other class (approx. log(0))
                log_probs = np.array([-np.inf, -np.inf])
                cls = np.argmax(y)
                log_probs[cls] = 0
                C[rejected_idx] = log_probs

        logger.info(f"LinearSVC was called {len(rejected_samples)/len(X)*100:.1f}% of the times.")

        # memory cleanup and return
        del rejected_samples
        del rejected_indices
        return np.array(C)

    def score(self, X, y):
        y_pred = self.predict(X)
        binary_acc_vector = [1 if y_gt_ == y_pred_ else 0 for (y_gt_, y_pred_) in zip(y, y_pred)]
        return sum(binary_acc_vector) / len(binary_acc_vector)
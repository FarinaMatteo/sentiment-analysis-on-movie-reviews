import time
import numpy as np
from utils.misc import get_basic_logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logger = get_basic_logger("TwoStageClassifier")

class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, neg_cls_min_confidence=0.79, pos_cls_min_confidence=0.81, second_stage_starting_point=None):
        assert 0 < neg_cls_min_confidence < 1, "The min confidence for the Negative Class must be within (0,1)"
        assert 0 < pos_cls_min_confidence < 1, "The min confidence for the Positive Class must be wihtin (0,1)"
        self.first_stage_clf = MultinomialNB()
        self.second_stage_clf = LinearSVC()
        self.neg_cls_min_confidence = neg_cls_min_confidence
        self.pos_cls_min_confidence = pos_cls_min_confidence
        self.second_stage_starting_point = second_stage_starting_point

    def hsplit_dataset(self, X):
        if self.second_stage_starting_point:
            X_first_stage = X[:, :self.second_stage_starting_point]
            X_second_stage = X[:, self.second_stage_starting_point:]
        else:
            X_first_stage = X
            X_second_stage = X
        return X_first_stage, X_second_stage

    def fit_first_stage(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting 1st Stage Classifier")
        self.first_stage_clf.fit(X, y)

    def fit_second_stage(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Fitting 2nd Stage Classifier")
        self.second_stage_clf.fit(X, y)

    def fit(self, X, y) -> None:
        in_time = time.time()
        X_first_stage, X_second_stage = self.hsplit_dataset(X)
        self.fit_first_stage(X_first_stage, y)
        self.fit_second_stage(X_second_stage, y)
        logger.info("Fitting finished successfully. Total time={:.2f}s".format(time.time()-in_time))

    def reject_option(self, y) -> bool:
        cls = np.argmax(y)
        if cls == 0 and y[cls] < self.neg_cls_min_confidence: # negative case
            return True
        if cls == 1 and y[cls] < self.pos_cls_min_confidence: # positive case
            return True
        return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        Y = []
        X_first_stage, X_second_stage = self.hsplit_dataset(X)
        C_first_stage = self.first_stage_clf.predict_proba(X_first_stage)
        rejected_samples, rejected_indices = [], []
        for i, cls_proba in enumerate(C_first_stage):
            if self.reject_option(cls_proba):
                rejected_sample = X_second_stage[i, :].squeeze()
                rejected_samples.append(rejected_sample)
                rejected_indices.append(i)
            else:
                y = np.argmax(cls_proba)
                Y.append(y)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples) > 0:
            rejected_samples = np.array(rejected_samples)
            Y_second_stage = self.second_stage_clf.predict(rejected_samples)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                Y.insert(rejected_idx, y)

        # memory cleanup and return
        del C_first_stage
        del rejected_samples
        del rejected_indices
        return np.array(Y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_first_stage, X_second_stage = self.hsplit_dataset(X)
        C = self.first_stage_clf.predict_proba(X_first_stage)
        rejected_samples, rejected_indices = [], []
        for i, c in enumerate(C):
            if self.reject_option(c):
                rejected_sample = X_second_stage[i, :].squeeze()
                rejected_samples.append(rejected_sample)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples) > 0:
            rejected_samples = np.array(rejected_samples)
            Y_second_stage = self.second_stage_clf.predict(rejected_samples)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a class probability of 1 to the class predicted by the SVM
                # 0 probability otherwise
                cls_probs = np.array([.0,.0])
                cls = np.argmax(y)
                cls_probs[cls] = 1.
                C[rejected_idx] = cls_probs

        # memory cleanup and return
        del rejected_samples
        del rejected_indices
        return np.array(C)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_first_stage, X_second_stage = self.hsplit_dataset(X)
        C = self.first_stage_clf.predict_log_proba(X_first_stage)
        rejected_samples, rejected_indices = [], []
        for i, c in enumerate(C):
            if self.reject_option(c):
                rejected_sample = X_second_stage[i, :].squeeze()
                rejected_samples.append(rejected_sample)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples) > 0:
            rejected_samples = np.array(rejected_samples)
            Y_second_stage = self.second_stage_clf.predict(rejected_samples)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a log-prob of 0 to the predicted class (log(1))
                # assign negative infinity to the other class (approx. log(0))
                log_probs = np.array([-np.inf, -np.inf])
                cls = np.argmax(y)
                log_probs[cls] = 0
                C[rejected_idx] = log_probs

        # memory cleanup and return
        del rejected_samples
        del rejected_indices
        return np.array(C)

    def score(self, X, y):
        y_pred = self.predict(X)
        onehot_accuracy = [1 if y_gt_ == y_pred_ else 0 for (y_gt_, y_pred_) in zip(y, y_pred)]
        return sum(onehot_accuracy) / len(onehot_accuracy)
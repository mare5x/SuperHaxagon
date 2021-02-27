import pathlib
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


class DAGGER:
    """Imitation learning: DAGGER algorithm"""
    
    MODEL_FPATH = "dagger.model"

    def __init__(self):
        self.model = RandomForestClassifier()

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        self.iteration = 0
        self.data_in_iteration = 0

        self.score_history = []

        self.SAMPLE_PROBABILITY = 0.2
        self.TEST_PERCENT = 0.25
        self.DATA_PER_ITERATION = 3000  # Gather this many data points each training iteration.
        
    @classmethod
    def load(cls):
        p = pathlib.Path(cls.MODEL_FPATH)
        if p.exists():
            print(f"Loading {cls} from {cls.MODEL_FPATH} ...")
            return joblib.load(cls.MODEL_FPATH)
        else:
            return cls()

    def write(self):
        joblib.dump(self, self.MODEL_FPATH)

    def train_model(self):
        X = np.array(X_train)
        y = np.array(y_train)
        self.model.fit(X, y)

    def store_experience(self, state, action):
        k = random.random()
        if k < self.SAMPLE_PROBABILITY:
            return False
        k = (k - self.SAMPLE_PROBABILITY) / (1.0 - self.SAMPLE_PROBABILITY)

        if k < self.TEST_PERCENT:
            self.X_test.append(state)
            self.y_test.append(action)
        else:
            self.X_train.append(state)
            self.y_train.append(state)
        
        self.data_in_iteration += 1
        if self.data_in_iteration % 100 == 0:
            print(f"Iteration [{self.iteration}]: data acquired {self.data_in_iteration}/{self.DATA_PER_ITERATION}")

        return True

    def get_action(self, state):
        # The expert is implemented on the C++ side ...
        # if self.iteration == 0:
        #     return expert action
        return self.model.predict(state)

    def evaluate_performance(self):
        X = np.array(self.X_test)
        y_true = np.array(self.y_test)
        y_pred = self.model.predict(X)
        return accuracy_score(y_true, y_pred)

    def on_episode_end(self, score):
        self.score_history.append(score)
        
        if self.data_in_iteration >= self.DATA_PER_ITERATION:
            print(f"Training iteration: {self.iteration}\n\tTrain: {len(self.X_train)} || Test: {len(self.X_test)}")
            self.train_model()
            
            perf = self.evaluate_performance()
            print(f"Iteration [{self.iteration}]: {perf:.2f}")

            self.iteration += 1
            self.data_in_iteration = 0

            self.write()



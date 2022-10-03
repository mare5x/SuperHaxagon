import pathlib
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
import joblib

from plot import plot_queue, moving_average


def plot(ax, data):
    ax.set_title("DAGGER score history")
    x = np.array(data) / 60  # Seconds
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Attempt number')
    ax.plot(x)
    ax.plot(moving_average(x, k=10))


class DAGGER:
    """Imitation learning: DAGGER algorithm using Random Forests for action classification."""
    
    MODEL_FPATH = "dagger_model.gz"
    DATA_FPATH = "dagger_data.gz"
    dump_data = True

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
        inst = cls()
        paths = [pathlib.Path(cls.MODEL_FPATH), pathlib.Path(cls.DATA_FPATH)]
        for p in paths:
            if p.exists():
                print(f"Loading {p} ...")
                data = joblib.load(p)
                for k, v in data.items():
                    setattr(inst, k, v)
        return inst

    def write(self):
        print(f"Writing to {self.MODEL_FPATH} ...")
        model = { 'model': self.model }
        data = { 
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'iteration': self.iteration,
            'data_in_iteration': self.data_in_iteration,
            'score_history': self.score_history
        }
        joblib.dump(model, self.MODEL_FPATH)
        if self.dump_data:
            joblib.dump(data, self.DATA_FPATH)

    def train_model(self):
        X = np.array(self.X_train)
        y = np.array(self.y_train)
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
            self.y_train.append(action)
        
        self.data_in_iteration += 1
        if self.data_in_iteration % 100 == 0:
            print(f"Iteration [{self.iteration}]: data acquired {self.data_in_iteration}/{self.DATA_PER_ITERATION}")

        return True

    def get_action(self, state, expert_action=None):
        # The expert is implemented on the C++ side. 
        # The expert action is passed along with the state from the 'server'.
        if expert_action is not None:
            self.store_experience(state, expert_action)
            if self.iteration == 0 and self.data_in_iteration < self.DATA_PER_ITERATION:
                return expert_action
        try:
            state = np.array(state).reshape(1, -1)  # Model requires 2D array
            return self.model.predict(state)[0]
        except NotFittedError:
            return 0

    def evaluate_performance(self):
        X = np.array(self.X_test)
        y_true = np.array(self.y_test)
        y_pred = self.model.predict(X)
        return accuracy_score(y_true, y_pred)

    def on_episode_end(self, score):
        self.score_history.append(score)
        plot_queue.put((plot, self.score_history))

        if self.data_in_iteration >= self.DATA_PER_ITERATION:
            print(f"Training iteration: {self.iteration}\n\tTrain: {len(self.X_train)} || Test: {len(self.X_test)}")
            self.train_model()
            
            perf = self.evaluate_performance()
            print(f"Iteration [{self.iteration}]: {perf:.2f}")

            self.iteration += 1
            self.data_in_iteration = 0

            self.write()

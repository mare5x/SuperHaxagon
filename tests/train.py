import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


def read_file(path):
    with open(path, "r", encoding="utf8") as f:
        n = int(f.readline())
        X = np.zeros((n, 15))
        y = np.zeros(n) 
        for i in range(n):
            X[i, :] = [float(x) for _ in range(7) for x in f.readline().split()]
            y[i] = float(f.readline())
        return X, y


X_train, y_train = read_file("training_data.txt")
X_test, y_test = read_file("validation_data.txt")


model = RandomForestClassifier()
# model = MLPClassifier(hidden_layer_sizes=(32, 32), solver="sgd", learning_rate="adaptive", learning_rate_init=0.01, max_iter=1000, shuffle=False, verbose=True, alpha=0.01)
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# print(model.feature_importances_)

# Results: 
# RandomForestClassifier: 0.97
# LogisticRegression: 0.51
# SVC: 0.68
# LinearSVC: 0.51
# GradientBoostingClassifier: 0.86
# MLPClassifier(hidden_layer_sizes=(32, 32), solver="sgd", learning_rate="adaptive", learning_rate_init=0.01, max_iter=1000, shuffle=False, verbose=True, alpha=0.01)
#   0.95

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, r2_score


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_model(name):
    if name == "RandomForest":
        return RandomForestClassifier()
    elif name == "SVM":
        return SVC()
    elif name == "LogisticRegression":
        return LogisticRegression()


def train_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    model.fit(X, y)
    return model, scores


def evaluate_model(model, X_test, y_test, problem):
    y_pred = model.predict(X_test)

    if problem == "Classification":
        return accuracy_score(y_test, y_pred)
    else:
        return r2_score(y_test, y_pred)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def get_model(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100)
    elif name == "SVM":
        return SVC(probability=True)
    elif name == "LogisticRegression":
        return LogisticRegression(max_iter=1000)


def train_model(model, X_train, y_train, k=5):
    scores = cross_val_score(model, X_train, y_train, cv=k)
    model.fit(X_train, y_train)
    return model, scores


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
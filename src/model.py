import os
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

def train_model(X, y):
    os.makedirs("models", exist_ok=True)  # Ensure directory exists
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    dump(clf, "models/troll_detector.joblib")
    return clf

def load_model():
    return load("models/troll_detector.joblib")

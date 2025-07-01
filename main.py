import pandas as pd
import numpy as np
from src.features import extract_features
from src.model import train_model, load_model
from src.evaluate import evaluate
from src.explain import explain_prediction

# Load JSON Lines dataset
json_path = "data/Dataset for Detection of Cyber-Trolls.json"
df = pd.read_json(json_path, lines=True)

print("Columns in dataset:", df.columns.tolist())
print("Sample row:\n", df.head(1))

# Extract label from annotation dict
def extract_label(annotation):
    # Defensive: check if annotation is a dict and has 'label'
    if isinstance(annotation, dict) and 'label' in annotation:
        # Some datasets use string labels, some int, adjust as needed
        val = annotation['label'][0]
        return int(val) if str(val).isdigit() else 1 if str(val) != '0' else 0
    return 0  # default to 0 (genuine) if missing

df['label'] = df['annotation'].apply(extract_label)

# Use the correct text column
text_col = 'content'

# Simulate user IDs (since original data lacks them)
df['user_id'] = df.index // 5  # Group every 5 tweets as one user (adjust as needed)

# Aggregate texts per user
user_groups = df.groupby("user_id")
user_features = []
labels = []
for user_id, group in user_groups:
    texts = group[text_col].tolist()
    features = extract_features(texts)
    user_features.append(list(features.values()))
    # Assign label: majority label in group
    labels.append(int(group['label'].mean() >= 0.5))

X = np.array(user_features)
y = np.array(labels)

# Stratified Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train labels distribution:", np.bincount(y_train))
print("Test labels distribution:", np.bincount(y_test))

# Train model
model = train_model(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
results = evaluate(y_test, y_pred)
print("Evaluation metrics:", results)

# Explain prediction
feature_names = ["mean_toxicity", "max_toxicity", "toxic_msg_ratio", "mean_sentiment", "min_sentiment", "num_msgs"]
explanation = explain_prediction(model, X_test, feature_names)
print("Feature Importances:", explanation)

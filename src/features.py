from src.toxicity import toxicity_score
from src.preprocessing import preprocess_text
from textblob import TextBlob

def extract_features(user_texts):
    n_texts = len(user_texts)
    toxicities = [toxicity_score(preprocess_text(t)) for t in user_texts]
    sentiments = [TextBlob(t).sentiment.polarity for t in user_texts]
    features = {
        "mean_toxicity": sum(toxicities) / n_texts,
        "max_toxicity": max(toxicities),
        "toxic_msg_ratio": sum([s > 0.2 for s in toxicities]) / n_texts,
        "mean_sentiment": sum(sentiments) / n_texts,
        "min_sentiment": min(sentiments),
        "num_msgs": n_texts,
    }
    return features

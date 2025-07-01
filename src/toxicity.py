TOXIC_WORDS = {"idiot", "stupid", "hate", "dumb", "fool", "moron", "trash", "loser"}

def toxicity_score(text):
    tokens = set(text.split())
    toxic_count = len(tokens & TOXIC_WORDS)
    return toxic_count / (len(tokens) + 1)

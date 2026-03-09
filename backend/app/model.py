# Temporary prediction, change to fit model
def predict_text(text: str) ->dict:
    text = text.lower()

    if "fake" in text or "hoax" in text:
        label = "misinformation"
        confidence = 0.85
    else:
        label = "reliable"
        confidence = 0.72

    return {
        "label": label,
        "confidence": confidence
    } 
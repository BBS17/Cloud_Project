from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

#This file is in charge of passing the user text to the ml model

# load saved model
model_path = "./misinformation_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval()

#parameters: user text
#return: label, confidence score
def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    #convert to percentage
    confidence = round(confidence * 100, 2)

    #change labels
    label_map = {
        0: "Truth",
        1: "Misinformation"
    }

    return {
        "label": label_map[predicted_class],
        "confidence": confidence
    }

#test
'''
if __name__ == "__main__":
    sample_text = "The clavicle is superior to the rib cage."
    result = predict_text(sample_text)
    print(result)
'''
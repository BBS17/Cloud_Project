import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import classification_report

# dataset https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

#import dataset
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

#add labels
fake_df["label"] = 1
true_df["label"] = 0

#merge datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Use titles only (short text)
df["text"] = df["title"]
df = df[["text", "label"]]

#clean dataset, drop missing values
df = df.dropna()

'''
#temp test
#Print results
print(df.head())
print("\nShape:", df.shape)
print("\nLabel distribution:\n", df["label"].value_counts())
# Temporary prediction, change to fit model
'''

#split dataset: training, validating, testing
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

'''
testing the split, 70/15/15
print("Train size:", len(train_texts))
print("Validation size:", len(val_texts))
print("Test size:", len(test_texts))
'''
#tokenize, distilbert doesn't accept raw text
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

#tokenize text
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128
)

'''
test tokenizer
print(train_texts[0])
print(train_encodings["input_ids"][0])
print(train_encodings["attention_mask"][0])
'''
#create dataset objects
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            item[key] = torch.tensor(val[idx])
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)
test_dataset = NewsDataset(test_encodings, test_labels)
'''
testing
print(train_dataset[0])
'''

#get pretrained base distilBert model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
    )

'''
testing
print(model)
'''

#using accuracy, change to different score late
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

#set training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2
)

#create trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

#train the model
trainer.train()

#Evaluate testing data
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(test_results)

#get predictions
predictions = trainer.predict(test_dataset)
#print(predictions)

#convert predictions to predicted classes
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

#print report
print(classification_report(true_labels, predicted_labels))

#save model
model.save_pretrained("./misinformation_model")
tokenizer.save_pretrained("./misinformation_model")


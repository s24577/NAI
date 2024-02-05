from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import database
from sklearn.metrics import f1_score, recall_score, precision_score

current_directory = os.path.dirname(os.path.abspath(__file__))

sample_sentences_path = os.path.join(current_directory, "..", "database", "sample_sentences.txt")

with open(sample_sentences_path, "r", encoding="utf-8") as file:
    test_sentences = [line.strip().split(": ")[1] for line in file.readlines()]

tokenizer = AutoTokenizer.from_pretrained("SharanSMenon/22-languages-bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("SharanSMenon/22-languages-bert-base-cased")


def predict(in_sentence):
    tokenized = tokenizer(in_sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**tokenized)
    return model.config.id2label[outputs.logits.argmax(dim=1).item()]


predict(test_sentences)
print(test_sentences)

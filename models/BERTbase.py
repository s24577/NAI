import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.calculator import calculate_metrics

tokenizer = AutoTokenizer.from_pretrained("SharanSMenon/22-languages-bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("SharanSMenon/22-languages-bert-base-cased")


def predict(sentence):
    tokenized = tokenizer(sentence, return_tensors="pt")
    outputs = model(**tokenized)
    return model.config.id2label[outputs.logits.argmax(dim=1).item()]


def model_bertbase(dataset_path):
    dataset = pd.read_csv(dataset_path)
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        sentence = row["Sentence"]
        actual_language = row["Predicted_language"]
        predicted_language = predict(sentence)

        precision, recall, f1 = calculate_metrics(predicted_language, actual_language)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    model1_precision = sum(all_precisions) / len(all_precisions)
    model1_recall = sum(all_recalls) / len(all_recalls)
    model1_f1 = sum(all_f1s) / len(all_f1s)
    return model1_precision, model1_recall, model1_f1

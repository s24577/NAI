import csv

from datasets import load_dataset


def convert_label(label_example_input):
    if label_example_input == 0:
        return "negative"
    elif label_example_input == 1:
        return "neutral"
    elif label_example_input == 2:
        return "positive"


# Załaduj zbiór danych
dataset = load_dataset("ihassan1/auditor-sentiment")

sentences_list = []
labels_list = []

for label_example, sentence_example in zip(dataset['train']['label'][:5], dataset['train']['sentence'][:5]):
    sentences_list.append({"Text": sentence_example})
    labels_list.append({"Label": convert_label(label_example)})

with open("dataset.csv", "w", newline="", encoding="utf-8") as csvfile:
    # Utwórz obiekt writer
    csvwriter = csv.writer(csvfile)

    # Dodaj nagłówki do pliku CSV
    csvwriter.writerow(["Predicted_sentiment", "Sentence"])

    # Przewiń przez przykłady
    for label_example1, sentence_example1 in zip(labels_list, sentences_list):
        # Zapisz do pliku CSV
        csvwriter.writerow([label_example1["Label"], sentence_example1["Text"]])

print("Wyniki zostały zapisane do pliku: dataset.csv")

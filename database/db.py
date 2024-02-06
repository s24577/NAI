import csv

from datasets import load_dataset

# Załaduj zbiór danych
dataset = load_dataset("papluca/language-identification")

# Wybierz 100 przykładów z różnych języków
languages_list = []
sentences_list = []

for language_example, sentence_example in zip(dataset['train']['labels'][:30], dataset['train']['text'][:30]):
    # if language_example not in used_languages:
    languages_list.append({"Language": language_example})
    sentences_list.append({"Text": sentence_example})

with open("dataset.csv", "w", newline="", encoding="utf-8") as csvfile:
    # Utwórz obiekt writer
    csvwriter = csv.writer(csvfile)

    # Dodaj nagłówki do pliku CSV
    csvwriter.writerow(["Predicted_language", "Sentence"])

    # Przewiń przez przykłady
    for language_example1, sentence_example1 in zip(languages_list, sentences_list):
        # Zapisz do pliku CSV
        csvwriter.writerow([language_example1["Language"], sentence_example1["Text"]])

print("Wyniki zostały zapisane do pliku: dataset.csv")

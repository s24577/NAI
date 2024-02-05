# Importujemy funkcję load_dataset z biblioteki datasets od Hugging Face.
from datasets import load_dataset

# Ładuje zbiór danych "papluca/language-identification" z Hugging Face Datasets.
dataset = load_dataset("papluca/language-identification")

# Wybieramy 100 różnych zdań, każde w innym języku
sample_sentences = []

# Iteruje przez pierwsze 100 przykładów
for example in dataset["train"]["text"][:100]:
    # Dodaje zdanie do listy sample_sentences
    sample_sentences.append(example)

# Wyświetl przykłady
for sentence in sample_sentences:
    print(f"Sentence: {sentence}")
# Zebrane wyniki zapisujemy do pliku
with open("sample_sentences.txt", "w", encoding="utf-8") as file:
    for sentence in sample_sentences:
        file.write(f"Sentence: {sentence}\n")

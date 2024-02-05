# Importujemy funkcję load_dataset z biblioteki datasets od Hugging Face.
from datasets import load_dataset

# Ładuje zbiór danych "papluca/language-identification" z Hugging Face Datasets.
dataset = load_dataset("papluca/language-identification")

# Wybieramy 100 różnych zdań, każde w innym w języku
# Tworzymy dwie puste listy, sample_sentences zawiera wybrane przykłądy zdań
sample_sentences = []
# language używamy do śledzenia, które języki dodano
languages = set()
# Iteruje przez pierwsze 100 przykładów
for example in dataset["train"]["text"][:100]:
    # Pobiera informację o języku w danym zdaniu
    language = example["language"]
    # Sprawdzamy czy owy język nie został dodany
    if language not in languages:
        # Dodaje zdanie do listy sample_sentences i dodaje język do zbioru languages.
        sample_sentences.append(example["sentence"])
        languages.add(language)
    # Przerywa pętle po uzyskaniu 100 przykładów
    if len(sample_sentences) == 100:
        break
# Zebranew wyniki zapisujemy do pliku
with open("sample_sentences.txt", "w", encoding="utf-8") as file:
    for sentence, language in zip(sample_sentences, languages):
        file.write(f"Language: {language}, Sentence: {sentence}\n")

# Importowanie niezbędnych bibliotek
from transformers import pipeline
import pandas as pd
from utils.calculator import calculate_metrics

# Inicjalizacja klasyfikatora sentymentu z wykorzystaniem modelu
distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=None
)

# Definicja funkcji model_distilbert
def model_distilbert(dataset_path):
    # Wczytanie zestawu danych
    dataset = pd.read_csv(dataset_path)
    # Inicjalizacja list do przechowywania rzeczywistych i przewidywanych sentymentów
    actual_sentiments, predicted_sentiments = [], []

    # Iteracja przez każdy wiersz w zestawie danych
    for index, row in dataset.iterrows():
        # Pobranie zdania i przewidywanego sentymentu z wiersza
        sentence = row["Sentence"]
        predicted_sentiment = row["Predicted_sentiment"]
        # Użycie klasyfikatora sentymentu do przewidzenia rzeczywistego sentymentu
        actual_sentiment = distilled_student_sentiment_classifier(sentence)[0][0]['label']

        # Dodanie rzeczywistego i przewidywanego sentymentu do odpowiednich list
        actual_sentiments.append(actual_sentiment)
        predicted_sentiments.append(predicted_sentiment)

    # Obliczenie metryk precyzji, pełności i F1
    precision, recall, f1 = calculate_metrics(actual_sentiments, predicted_sentiments)

    # Zwrócenie metryk
    return precision, recall, f1
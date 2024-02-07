# Importowanie niezbędnych bibliotek
import pandas as pd
from transformers import pipeline, logging
from utils.calculator import calculate_metrics

# Ustawienie poziomu logowania na błąd
logging.set_verbosity_error()

# Inicjalizacja zadania analizy sentymentu z wykorzystaniem modelu i tokenizera
sentiment_task = pipeline("sentiment-analysis",
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                          tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Definicja funkcji model_twitter_robert
def model_twitter_robert(dataset_path):
    # Wczytanie zestawu danych
    dataset = pd.read_csv(dataset_path)
    # Inicjalizacja list do przechowywania rzeczywistych i przewidywanych sentymentów
    actual_sentiments, predicted_sentiments = [], []

    # Iteracja przez każdy wiersz w zestawie danych
    for index, row in dataset.iterrows():
        # Pobranie zdania i przewidywanego sentymentu z wiersza
        sentence = row["Sentence"]
        predicted_sentiment = row["Predicted_sentiment"]
        # Użycie zadania analizy sentymentu do przewidzenia rzeczywistego sentymentu
        actual_sentiment = sentiment_task(sentence)[0]['label']

        # Dodanie rzeczywistego i przewidywanego sentymentu do odpowiednich list
        actual_sentiments.append(actual_sentiment)
        predicted_sentiments.append(predicted_sentiment)

    # Obliczenie metryk precyzji, pełności i F1
    precision, recall, f1 = calculate_metrics(actual_sentiments, predicted_sentiments)

    # Zwrócenie metryk
    return precision, recall, f1
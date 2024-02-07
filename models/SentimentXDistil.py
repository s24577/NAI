from transformers import pipeline
import pandas as pd

from utils.calculator import calculate_metrics

# Inicjalizacja pipeline'u do analizy sentymentu
sentiment_classifier = pipeline("sentiment-analysis",
                                model="hakonmh/sentiment-xdistil-uncased",
                                tokenizer="hakonmh/sentiment-xdistil-uncased")


# Input dataset_path to ścieżka do pliku z datasetem
def model_sentiment_xdistil(dataset_path):
    # Wczytanie datasetu
    dataset = pd.read_csv(dataset_path)
    # Inicjalizacja listy przechowywującej wszystkie wyniki, które zwróci model
    all_predictions = []

    # Iteracja przez dataset
    for index, row in dataset.iterrows():
        # Treśc pobrana z datasetu, która będzie poddawana analizie
        sentence = row["Sentence"]
        # Przewidywana predykcja pobrana z datasetu
        actual_sentiment = row["Predicted_sentiment"]
        # Wykonanie predykcji przez model
        predicted_sentiment = sentiment_classifier(sentence)
        # Wyciągnięcie sentymentu z wyniki
        predicted_label = predicted_sentiment[0]['label'].lower()

        # Dodanie wyniku zwróconego przez model do listy ze wszystkimi wynikami
        all_predictions.append((predicted_label, actual_sentiment))

    # Wyliczenie średnich: precision, recall oraz f1 poprzez funckję calculate_metrics
    average_precision, average_recall, average_f1 = calculate_metrics(all_predictions)

    return average_precision, average_recall, average_f1

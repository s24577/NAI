import pandas as pd
from transformers import pipeline

from utils.calculator import calculate_metrics

sentiment_task = pipeline("sentiment-analysis",
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                          tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")


def model_twitter_robert(dataset_path):
    dataset = pd.read_csv(dataset_path)
    all_predictions = []

    for index, row in dataset.iterrows():
        sentence = row["Sentence"]
        actual_sentiment = row["Predicted_sentiment"]
        predicted_sentiment = sentiment_task(sentence)
        predicted_label = predicted_sentiment[0]['label']

        all_predictions.append((predicted_label, actual_sentiment))

    average_precision, average_recall, average_f1 = calculate_metrics(all_predictions)

    return average_precision, average_recall, average_f1
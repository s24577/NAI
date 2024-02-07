import pandas as pd
from transformers import pipeline

from utils.calculator import calculate_metrics

sentiment_task = pipeline("sentiment-analysis",
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                          tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")


def model_twitter_robert(dataset_path):
    dataset = pd.read_csv(dataset_path)
    actual_sentiments, predicted_sentiments = [], []

    for index, row in dataset.iterrows():
        sentence = row["Sentence"]
        predicted_sentiment = row["Predicted_sentiment"]
        actual_sentiment = sentiment_task(sentence)[0]['label']

        actual_sentiments.append(actual_sentiment)
        predicted_sentiments.append(predicted_sentiment)

    precision, recall, f1 = calculate_metrics(actual_sentiments, predicted_sentiments)

    return precision, recall, f1

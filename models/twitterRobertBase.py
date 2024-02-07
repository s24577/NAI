import pandas as pd
from transformers import pipeline, RobertaForSequenceClassification, RobertaTokenizer

from utils.calculator import calculate_metrics

tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_task = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)


def model_twitter_robert(dataset_path):
    dataset = pd.read_csv(dataset_path)
    all_predictions = []

    for index, row in dataset.iterrows():
        sentence = row["Sentence"]
        actual_sentiment = row["Predicted_sentiment"]
        predicted_sentiment = sentiment_task(sentence)
        predicted_label = predicted_sentiment[0]['label']
        print(predicted_label)

        all_predictions.append((predicted_label, actual_sentiment))

    average_precision, average_recall, average_f1 = calculate_metrics(all_predictions)

    return average_precision, average_recall, average_f1

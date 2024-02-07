from transformers import pipeline
import pandas as pd

from utils.calculator import calculate_metrics

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=None
)


def model_distilbert(dataset_path):
    dataset = pd.read_csv(dataset_path)
    actual_sentiments, predicted_sentiments = [], []

    for index, row in dataset.iterrows():
        sentence = row["Sentence"]
        predicted_sentiment = row["Predicted_sentiment"]
        # Zakładam, że 'actual_sentiment' jest prawdziwą etykietą, a nie przewidywaną
        actual_sentiment = distilled_student_sentiment_classifier(sentence)[0][0]['label']

        actual_sentiments.append(actual_sentiment)
        predicted_sentiments.append(predicted_sentiment)

    precision, recall, f1 = calculate_metrics(actual_sentiments, predicted_sentiments)

    return precision, recall, f1

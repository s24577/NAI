from transformers import pipeline
import pandas as pd

from utils.calculator import calculate_metrics

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=None
)


def model_distilbert(dataset_path):
    dataset = pd.read_csv(dataset_path)
    all_predictions = []

    for index, row in dataset.iterrows():
        sentence = row["Sentence"]
        actual_sentiment = row["Predicted_sentiment"]
        predicted_sentiment = distilled_student_sentiment_classifier(sentence)
        predicted_label = predicted_sentiment[0][0]['label']
        print(predicted_sentiment[0][0]['label'])

        all_predictions.append((predicted_label, actual_sentiment))

    average_precision, average_recall, average_f1 = calculate_metrics(all_predictions)

    return average_precision, average_recall, average_f1

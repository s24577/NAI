from transformers import pipeline
import pandas as pd

from utils.calculator import calculate_metrics

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=None
)


def model_distilbert(dataset_path):
    dataset = pd.read_csv(dataset_path)
    all_precisions, all_recalls, all_f1s = [], [], []

    for index, row in dataset.iterrows():
        sentence = row["Sentence"]
        actual_sentiment = row["Predicted_sentiment"]
        predicted_sentiment = distilled_student_sentiment_classifier(sentence)
        print(predicted_sentiment)
    #     precision, recall, f1 = calculate_metrics(predicted_sentiment, actual_sentiment)
    #
    #     all_precisions.append(precision)
    #     all_recalls.append(recall)
    #     all_f1s.append(f1)
    #
    # model1_precision = sum(all_precisions) / len(all_precisions)
    # model1_recall = sum(all_recalls) / len(all_recalls)
    # model1_f1 = sum(all_f1s) / len(all_f1s)
    # return model1_precision, model1_recall, model1_f1
    return

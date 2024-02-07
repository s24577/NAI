# from rouge import Rouge
#
#
# def calculate_metrics(predicted_sentiment, actual_sentiment):
#     rouge = Rouge()
#     scores = rouge.get_scores(predicted_sentiment, actual_sentiment)
#
#     return scores[0]['rouge-l']['p'], scores[0]['rouge-l']['r'], scores[0]['rouge-l']['f']

from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(predictions):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for predicted_sentiment, actual_sentiment in predictions:
        precision = precision_score([actual_sentiment], [predicted_sentiment], average='weighted', zero_division=1)
        recall = recall_score([actual_sentiment], [predicted_sentiment], average='weighted', zero_division=1)
        f1 = f1_score([actual_sentiment], [predicted_sentiment], average='weighted', zero_division=1)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    return average_precision, average_recall, average_f1

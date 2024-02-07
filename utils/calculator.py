from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(actual_sentiment, predicted_sentiment):
    precision = precision_score(actual_sentiment, predicted_sentiment, average='weighted')
    recall = recall_score(actual_sentiment, predicted_sentiment, average='weighted')
    f1 = f1_score(actual_sentiment, predicted_sentiment, average='weighted')

    return precision, recall, f1

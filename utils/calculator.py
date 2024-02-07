from sklearn.metrics import precision_score, recall_score, f1_score


# Input: predictions to predykcje zwrócone przez dany model
def calculate_metrics(predictions):
    # Inicjalizacja list przechowywujących wyniki: precision, recall oraz f1 dla predykcji danego modelu
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Iteracja przez predykcje poprane z inputu
    for predicted_sentiment, actual_sentiment in predictions:
        # Obliczanie precision, recall oraz f1 dla każdej predykcji dzięki bibliotece sklearn.metrics
        precision = precision_score([actual_sentiment], [predicted_sentiment], average='weighted', zero_division=1)
        recall = recall_score([actual_sentiment], [predicted_sentiment], average='weighted', zero_division=1)
        f1 = f1_score([actual_sentiment], [predicted_sentiment], average='weighted', zero_division=1)

        # Dodawanie precision, recall oraz f1 do list przechowujących wyniki dla wszystkich predykcji
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Obliczanie średnich wyników precision, recall oraz f1
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)
    average_f1 = sum(f1_scores) / len(f1_scores)

    return average_precision, average_recall, average_f1

def print_results(model_name, precision, recall, f1):
    rounded_precision = "{:.3f}".format(precision)
    rounded_recall = "{:.3f}".format(recall)
    rounded_f1 = "{:.3f}".format(f1)

    print(f"{model_name} - Metrics:")
    print(f"Precision: {rounded_precision}, Recall: {rounded_recall}, F1: {rounded_f1}\n")


def print_all_results(model1_metrics, model2_metrics, model3_metrics):
    print("Models Metrics:\n")
    print_results("Model 1 - lxyuan/distilbert-base-multilingual-cased-sentiments-student", *model1_metrics)
    print_results("Model 2 - hakonmh/sentiment-xdistil-uncased", *model2_metrics)
    print_results("Model 3 - cardiffnlp/twitter-roberta-base-sentiment-latest", *model3_metrics)

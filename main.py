from models.DistilbertBase import model_distilbert
from models.SentimentXDistil import model_sentiment_xdistil
from utils.printer import print_all_results

if __name__ == "__main__":
    dataset_path = "database/dataset.csv"
    print("First model:...")
    model1_metrics = (model1_precision, model1_recall, model1_f1) = model_distilbert(dataset_path)

    print("Second model:...")
    model2_metrics = (model2_precision, model2_recall, model2_f1) = model_sentiment_xdistil(dataset_path)

    print_all_results(model1_metrics, model2_metrics)



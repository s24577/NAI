import models.TwitterRobertBase
from models.DistilbertBase import model_distilbert
from models.SentimentXDistil import model_sentiment_xdistil
from models.TwitterRobertBase import model_twitter_robert
from utils.printer import print_all_results

if __name__ == "__main__":
    dataset_path = "database/dataset.csv"
    model1_metrics = (model1_precision, model1_recall, model1_f1) = model_distilbert(dataset_path)
    model2_metrics = (model2_precision, model2_recall, model2_f1) = model_sentiment_xdistil(dataset_path)
    model3_metrics = (model3_precision, model3_recall, model3_f1) = model_twitter_robert(dataset_path)

    print_all_results(model1_metrics, model2_metrics, model3_metrics)

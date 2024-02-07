from models.DistilbertBase import model_distilbert
from models.twitterRobertBase import model_twitter_robert
from utils.printer import print_all_results

if __name__ == "__main__":
    dataset_path = "database/dataset.csv"
    print("First model:...")
    model1_metrics = (model1_precision, model1_recall, model1_f1) = model_distilbert(dataset_path)
    print("Third model:...")
    model3_metrics = (model3_precision, model3_recall, model3_f1) = model_twitter_robert(dataset_path)
    print_all_results(model1_metrics, model3_metrics)

from models.DistilbertBase import model_bertbase
from utils.printer import print_all_results

if __name__ == "__main__":
    dataset_path = "database/dataset.csv"
    print("First model:...")
    model1_metrics = (model1_precision, model1_recall, model1_f1) = model_bertbase(dataset_path)
    print_all_results(model1_metrics)


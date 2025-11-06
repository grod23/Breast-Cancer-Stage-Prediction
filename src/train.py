from breast_mri_dataset.dataset_utils import DataUtils
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.metrics import confusion_matrix


print(f'Device Available: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train:
    def __init__(self):
        self.training_logs = []
        self.validation_logs = []
        self.data_utils = DataUtils()
        self.training_loader, self.testing_loader = self.data_utils.create_dataloaders()

    def show_fit(self, metric:str, metric_logs:list, epochs:int):
        plt.figure(figsize=(10, 10))
        plt.plot(metric_logs, c='b', label=metric)
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.show()

        return metric_logs


    def find_best_fit(self, model, X_train, X_test, y_train, y_test):
        accuracy_logs = []
        precision_logs = []
        recall_logs = []
        f1_logs = []

        for n in range(100):
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_test)
            # Evaluation Metrics
            accuracy = accuracy_score(y_test,y_predicted)
            precision = precision_score(y_test, y_predicted, average='macro')
            recall = recall_score(y_test, y_predicted, average='macro')
            f1 = f1_score(y_test, y_predicted, average='macro')
            # Log metrics
            accuracy_logs.append(accuracy)
            precision_logs.append(precision)
            recall_logs.append(recall)
            f1_logs.append(f1)

        metric_logs = self.show_fit('Accuracy', accuracy_logs, len(accuracy_logs))

        max_metric = max(metric_logs)
        best_fit = metric_logs.index(max_metric)

        return best_fit


    def random_forest(self):
        X_train, X_test, y_train, y_test = self.data_utils.get_train_split()

        rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30)
        rfc.fit(X_train, y_train)
        y_predicted = rfc.predict(X_test)

        # Evaluation Metrics
        cm = confusion_matrix(y_predicted, y_test)
        report = classification_report(y_predicted, y_test)
        print(report)

        # Classification Matrix Heat Map
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()




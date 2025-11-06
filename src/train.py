from breast_mri_dataset.dataset_utils import DataUtils
from breast_mri_dataset.skl_dataset import SKL_Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.metrics import confusion_matrix

import sys

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
        # Train Test Split
        X_train, X_test, y_train, y_test = self.data_utils.get_train_split()
        # SKLearn Dataset Instances
        train_dataset = SKL_Dataset(X_train, y_train)
        test_dataset = SKL_Dataset(X_test, y_test)
        # Prepare data for SKLearn model input
        X_train, y_train = train_dataset.model_prep()
        X_test, y_test = test_dataset.model_prep()

        rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30, class_weight='balanced')
        multi_rfc = MultiOutputClassifier(rfc)
        multi_rfc.fit(X_train, y_train)
        y_pred = multi_rfc.predict(X_test)
        # Split predictions for T, N, M
        label_names = ['T', 'N', 'M']
        for i, label in enumerate(label_names):
            print(f"\nClassification report for {label}-stage:")
            print(classification_report(y_test[:, i], y_pred[:, i]))

            cm = confusion_matrix(y_test[:, i], y_pred[:, i])
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix: {label}-stage')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

        # rfc_T = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30, class_weight='balanced')
        # rfc_N = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30, class_weight='balanced')
        # rfc_M = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30, class_weight='balanced')
        #
        # # Get distinct tumor stage labels(must be integers)
        # t_labels_train = np.rint(y_train[:, 0]).astype(int)  # all T labels
        # n_labels_train = np.rint(y_train[:, 1]).astype(int)  # all N labels
        # m_labels_train = np.rint(y_train[:, 2]).astype(int)  # all M labels
        # t_labels_test = np.rint(y_test[:, 0]).astype(int)
        # n_labels_test = np.rint(y_test[:, 1]).astype(int)
        # m_labels_test = np.rint(y_test[:, 2]).astype(int)
        #
        # # Fit each label separately
        # rfc_T.fit(X_train, t_labels_train)
        # rfc_N.fit(X_train, n_labels_train)
        # rfc_M.fit(X_train, m_labels_train)
        #
        # # rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30)
        # # rfc.fit(X_train, y_train)
        #
        # # Get each label predictions
        # y_predicted_T = rfc_T.predict(X_test)
        # y_predicted_N = rfc_N.predict(X_test)
        # y_predicted_M = rfc_M.predict(X_test)
        #
        # # Option B: Other models (need MultiOutputClassifier wrapper)
        # # model = MultiOutputClassifier(SVC(kernel='rbf'))
        # # model.fit(X_train, y_train)
        #
        # y_predicted = [y_predicted_T, y_predicted_N, y_predicted_M]
        # y_true = [t_labels_test, n_labels_test, m_labels_test]
        # # Evaluation Metrics
        # for index, prediction in enumerate(y_predicted):
        #     report = classification_report(y_true[index], prediction)
        #     print(report)
        #     cm = confusion_matrix(y_true[index], prediction)
        #     # Classification Matrix Heat Map
        #     plt.figure(figsize=(10, 10))
        #     sns.heatmap(cm, annot=True, cmap='Blues', cbar=True)
        #     plt.title(f'Confusion Matrix: {index}')
        #     plt.xlabel('Predicted Label')
        #     plt.ylabel('True Label')
        #     plt.show()




from sklearn.model_selection import train_test_split
import joblib
import sys

def load_sequences_dict():
    # Load
    sequences = joblib.load("sequence_data.joblib")
    return sequences

def get_train_split():
    sequences = load_sequences_dict()
    # X
    patient_ids = [patient['patient_id'] for patient in sequences]
    print(patient_ids)
    # y
    labels = [patient['label'] for patient in sequences]
    print(len(patient_ids))
    print(len(labels))

    # Train Test Split
    X_train_ids, X_test_ids, y_train, y_test = train_test_split(patient_ids, labels, test_size=0.20, random_state=30)

    print(len(X_train_ids))
    print(len(y_train))
    print(len(X_test_ids))
    print(len(y_test))

    # Currently list of patient ids
    # Want list of sequences
    # Filter sequences for training and testing
    X_train = []
    for sequence_id in X_train_ids:
        X_train.append(sequence_id)


    X_train = [seq for seq in sequences if seq['patient_id'] in X_train_ids]
    X_test = [seq for seq in sequences if seq['patient_id'] in X_test_ids]
    y_train = [seq['label'] for seq in X_train]
    y_test = [seq['label'] for seq in X_test]




    # Convert Dictionaries to list of sequences


    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    print(y_test)

get_train_split()
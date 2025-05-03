import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


import os
import csv
import pickle

def load_model(pkl_path):
    with open(pkl_path, 'rb') as file:
        model = pickle.load(file)
        return model

def save_model(model, root_path, model_name):
    pkl_path = f"{root_path}{model_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)

def parse_dataset(ml, stories, feature, stride, grade):
    model_name =f"{ml} "
    
    if stride == 0:
        model_name += f"All N-Grams"
    elif stride == -1:
        model_name += f"Sentence Fragments"
    elif stride in [1, 2, 3, 100]:
        model_name += f"N = {stride}"
    else:
        print(f"Invalid stride length: {stride}")
        return
    
    if feature == "B":
        feature_set = ['sent_len', 'word_len', 'syll_num', 'poly_num', 'noun_tr', 'verb_tr', 'type_tr', 'lex_density', 'lex_foreign']
        model_name += " [Trad + Lex]"
    elif feature == "T":
        feature_set = ['sent_len', 'word_len', 'syll_num', 'poly_num']
        model_name += " [Trad]"
    elif feature == "L":
        feature_set = ['noun_tr', 'verb_tr', 'type_tr', 'lex_density', 'lex_foreign']
        model_name += " [Lex]"
    else:
        print("Invalid feature set selected. Please choose 'B', 'T', or 'L'.")
        return
    
    # if stories == "full":
    #     X_train, y_train, X_test, y_test = load_one(stride, feature_set, grade)
    # elif stories == "split":
    #     train_dataset, test_dataset = load_dataset(stride, grade)
    # else:
    #     print("Invalid stories option. Please choose 'full' or 'split'.")
    #     return
    
    if stories == "split":
        train_dataset, test_dataset = load_dataset(stride, grade)
    else:
        print("Invalid stories option. Please choose 'split'.")
        return    
    
    return train_dataset, test_dataset, feature_set, model_name

def filter_dataset(dataset, stride_length):
    # Return sentence fragments
    if stride_length == -1:
        filtered_dataset = dataset[(dataset['stride_len'] > 3) & (dataset['stride_len'] < 100)]
    # Do not filter
    elif stride_length == 0:
        return dataset
    else:
        filtered_dataset = dataset[dataset['stride_len'] == stride_length]
    
    return filtered_dataset

def load_one(stride_length, feature_set, grade):
    dataset_source = 'tables/allbooks_dataset.csv'
    
    df= pd.read_csv(dataset_source)
    df = filter_dataset(df, stride_length)
    
    df = df[df['text_num'] != 18]
    df = df.drop(columns=['word_num'])
    df['target'] = (df['grade_level'] == grade).astype(int)
    
    # X = df.drop(columns=feature_set)
    X = df[feature_set]
    y = df['grade_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, y_train, X_test, y_test
    

def load_dataset(stride_length, grade):
    training_source = 'tables/dataset.csv'
    testing_source = 'tables/dataset_testing.csv'
    
    train_dataset= pd.read_csv(training_source)
    train_dataset = filter_dataset(train_dataset, stride_length)
    train_dataset = undersample_dataset(train_dataset, grade)
    
    train_dataset = train_dataset[train_dataset['text_num'] != 18]
    train_dataset = train_dataset.drop(columns=['word_num'])

    train_dataset['target'] = (train_dataset['grade_level'] == grade).astype(int)

    # Load testing dataset
    test_dataset = pd.read_csv(testing_source)
    test_dataset = filter_dataset(test_dataset, stride_length)
    test_dataset = undersample_dataset(test_dataset, grade)

    test_dataset = test_dataset.drop(columns=['word_num'])

    test_dataset['target'] = (test_dataset['grade_level'] == grade).astype(int)
    
    return train_dataset, test_dataset

def split_dataset(train_dataset, test_dataset, feature_set):
    X_train = train_dataset[feature_set]
    y_train = train_dataset['target']
    
    X_test = test_dataset[feature_set]
    y_test = test_dataset['target']
    
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train, y_train, X_test, y_test
    

def fold_dataset(train_dataset, test_dataset, feature_set, k, fold_index):   
    df = train_dataset

    # Create stratified K folds on training set
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(df[feature_set], df['target'])):
        if i == fold_index:
            train_df = df.iloc[train_idx]
            break

    X_train = train_df[feature_set]
    y_train = train_df['target']

    test_df = test_dataset

    X_test = test_df[feature_set]
    y_test = test_df['target']

    return X_train, y_train, X_test, y_test


def undersample_dataset(df, grade):
    # Split the dataset into matching and non-matching groups
    matching_grade = df[df['grade_level'] == grade]
    non_matching_grade = df[df['grade_level'] != grade]
    
    # Find the smaller group size
    min_count = min(len(matching_grade), len(non_matching_grade))
    
    # Sample from both groups to ensure equal size
    matching_sampled = matching_grade.sample(min_count, random_state=42)
    non_matching_sampled = non_matching_grade.sample(min_count, random_state=42)
    
    # Combine the two groups
    undersampled_df = pd.concat([matching_sampled, non_matching_sampled]).reset_index(drop=True)
    
    return undersampled_df
    
def model_performance(model, X_test, y_test):
    # y_pred = model.predict(X_test)    
    # score = accuracy_score(y_test, y_pred)
    # return score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Binarize true labels for multi-class ROC-AUC
    class_labels = [0, 1]
    y_test_bin = label_binarize(y_test, classes=class_labels)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

def evaluate_model(model, model_name, X_test, y_test, machine, grade, feature, stride):
    # Predictions & Probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Use probabilities for the positive class (class 1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)  # Use 1D y_test and probabilities for class 1

    print(f"{machine} | {grade} | {feature} | {stride}")  
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save metrics to CSV
    model_details = {
        "machine": machine,
        "grade": grade,
        "feature": feature,
        "stride": stride,
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1_score": f"{f1:.4f}",
        "roc_auc": f"{roc_auc:.4f}"
    }
    
    csv_path = "models/performance_records/average_metrics.csv"
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as dataset:
        fieldnames = ["machine", "grade", "feature", "stride", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
        writer = csv.DictWriter(dataset, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(model_details)

    model_id = f"{machine}_{grade}_{feature}_{stride}"
    plot_path = f"models/performance_records/plots"
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_path, f"{model_id}_cm.png"))
    plt.close(fig)

def export_plot(model, model_name, model_id, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    class_labels = [0, 1]
    
    plot_path = f"models/performance_records/plots"
    
     # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_path, f"{model_id}_cm.png"))
    plt.close(fig)

    # Feature Importance Plot
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importances[indices], y=np.array(X_test.columns)[indices], ax=ax)
        ax.set_title(f"{model_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_path, f"{model_id}_feat.png"))
        plt.close(fig)

def export_metrics(metrics, machine, grade, feature, stride):
    csv_path="models/performance_records/average_metrics.csv"
    
    averages = {key: np.mean(values) for key, values in metrics.items()}
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if the file already exists
    file_exists = os.path.exists(csv_path)
    
    # Write metrics to the CSV file
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["machine", "grade", "feature", "stride", "accuracy", "precision", "recall", "f1", "roc_auc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the row with model details and average metrics
        writer.writerow({
            "machine": machine,
            "grade": grade,
            "feature": feature,
            "stride": stride,
            "accuracy": f"{averages['accuracy']:.4f}",
            "precision": f"{averages['precision']:.4f}",
            "recall": f"{averages['recall']:.4f}",
            "f1": f"{averages['f1']:.4f}",
            "roc_auc": f"{averages['roc_auc']:.4f}"
        })
    
    print(f"Average metrics saved to {csv_path}")

def cross_validation_scores(model, X, y, cv=5):
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc'
    }

    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    print("Cross-Validation Results ({}-fold):".format(cv))
    for metric in scoring.keys():
        mean_score = results[f'test_{metric}'].mean()
        print(f"{metric.capitalize()}: {mean_score:.3f}")

    return results

def export_details(pkl_path):    
    file_name = pkl_path.split("/")[-1]
    file_name = file_name.replace(".pkl", "")
    parts = file_name.split("_")

    machine = parts[0]
    stories = parts[1]
    feature = parts[2]
    stride = int(parts[3])
    
    # print(pkl_path)
    # print(f"stpries: {stories}")
    # print(f"feature: {feature}")
    # print(f"stride: {stride}")
    
    model = load_model(pkl_path)
    X_train, y_train, X_test, y_test, model_name = parse_dataset("lr", stories, feature, stride)
    cross_validation_scores(model, X_train, y_train)
    
    # evaluate_model(model, model_name, X_test, y_test, machine, stories, feature, stride) 

def main():
    base_path = "models/svm_models/untuned/"
    
    # Walk through all files in the base_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_path = os.path.join(root, file)
                print(f"\nProcessing file: {pkl_path}")
                export_details(pkl_path)
                # print(pkl_path)
                
# main()
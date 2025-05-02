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

import os
import csv
import pickle

def load_model(pkl_path):
    with open(pkl_path, 'rb') as file:
        model = pickle.load(file)
        return model

def parse_dataset(ml, grade, feature, stride):
    model_name =f"{ml}"
    
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
    #     X_train, y_train, X_test, y_test = load_one(stride, feature_set)
    # elif stories == "split":
    #     X_train, y_train, X_test, y_test = load_dataset(stride, feature_set, grade)
    # else:
    #     print("Invalid stories option. Please choose 'full' or 'split'.")
    #     return
    
    X_train, y_train, X_test, y_test = load_dataset(stride, feature_set, grade)
    
    # Scale to standardize the column values
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train, y_train, X_test, y_test, model_name

# Filter dataset based on stride length
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

# def load_one(stride_length, feature_set):
#     dataset_source = 'tables/allbooks_dataset.csv'
    
#     df= pd.read_csv(dataset_source)
#     df = filter_dataset(df, stride_length)
    
#     df = df[df['text_num'] != 18]
#     df = df.drop(columns=['word_num'])
    
#     # X = df.drop(columns=feature_set)
#     X = df[feature_set]
#     y = df['grade_level']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
#     return X_train, y_train, X_test, y_test
    

def load_dataset(stride_length, feature_set, grade):
    training_source = 'tables/dataset.csv'
    testing_source = 'tables/dataset_testing.csv'
    
    train_dataset= pd.read_csv(training_source)
    
    # Filter according to grade and stride
    train_dataset = filter_dataset(train_dataset, stride_length)
    train_dataset = train_dataset[train_dataset['grade_level'] == grade]
    
    # train_dataset = undersample_dataset(train_dataset)
    
    # Filter out rows with from 18.txt and drop empty column word_len
    train_dataset = train_dataset[train_dataset['text_num'] != 18]
    train_dataset = train_dataset.drop(columns=['word_num'])

    # Split the dataset into features and target variable
    X_train = train_dataset[feature_set]
    y_train = train_dataset['grade_level']

    # Load testing dataset
    test_dataset = pd.read_csv(testing_source)
    test_dataset = filter_dataset(test_dataset, stride_length)
    # test_dataset = test_dataset[test_dataset['grade_level'] == grade]
        
    # test_dataset = undersample_dataset(test_dataset)

    test_dataset = test_dataset.drop(columns=['word_num'])

    # X_test = test_dataset.drop(columns=feature_set)
    X_test = test_dataset[feature_set]
    y_test = test_dataset['grade_level']
    
    return X_train, y_train, X_test, y_test

def undersample_dataset(df):
    # Find the minimum count of entries across all grade levels
    min_count = df['grade_level'].value_counts().min()
    
    # Group by grade level and sample min_count entries from each group
    undersampled_df = df.groupby('grade_level').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    return undersampled_df

def print_scores(model, X_test, y_test):    
    y_pred = model.predict(X_test)
    
    # Check if the model supports probability predictions
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    # Binarize true labels for multi-class ROC-AUC
    class_labels = [1, 2, 3, 4, 5, 6]
    y_test_bin = label_binarize(y_test, classes=class_labels)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

def evaluate_model(model, model_name, X_test, y_test, machine, stories, feature, stride):
    # Predictions & Probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Binarize true labels for multi-class ROC-AUC
    class_labels = [1, 2, 3, 4, 5, 6]
    y_test_bin = label_binarize(y_test, classes=class_labels)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    
    print(f"{machine} | {stories} | {feature} | {stride}")  
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save metrics to CSV
    model_details = {
        "machine": machine,
        "stories": stories,
        "feature": feature,
        "stride": stride,
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1_score": f"{f1:.4f}",
        "roc_auc": f"{roc_auc:.4f}"
    }
    
    csv_path = "models/performance_records/models_analysis.csv"
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as dataset:
        fieldnames = ["machine", "stories", "feature", "stride", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
        writer = csv.DictWriter(dataset, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(model_details)

def export_plots(model, model_name, X_test, y_test, machine, stories, feature, stride):
    model_id = f"{machine}_{stories}_{feature}_{stride}"
    plot_path = f"models/performance_records/plots"
    
    y_pred = model.predict(X_test)
    class_labels = [1, 2, 3, 4, 5, 6]
    
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

def cross_validation_scores(model, X_train, y_train, machine, stories, feature, stride):
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc_ovr' if len(np.unique(y_train)) > 2 else 'roc_auc'
    }

    # Perform cross-validation
    results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, return_train_score=False)

    print("Cross-Validation Results (5-fold):")
    mean_scores = {}
    for metric in scoring.keys():
        mean_score = results[f'test_{metric}'].mean()
        mean_scores[metric] = mean_score
        print(f"{metric.capitalize()}: {mean_score:.3f}")

    # Prepare data for CSV export
    csv_path = "models/performance_records/cross_validation.csv"
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as dataset:
        fieldnames = ["machine", "stories", "feature", "stride", "accuracy", "precision", "recall", "f1", "roc_auc"]
        writer = csv.DictWriter(dataset, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "machine": machine,
            "stories": stories,
            "feature": feature,
            "stride": stride,
            "accuracy": f"{mean_scores['accuracy']:.4f}",
            "precision": f"{mean_scores['precision']:.4f}",
            "recall": f"{mean_scores['recall']:.4f}",
            "f1": f"{mean_scores['f1']:.4f}",
            "roc_auc": f"{mean_scores['roc_auc']:.4f}"
        })

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
    X_train, y_train, X_test, y_test, model_name = parse_dataset(machine, stories, feature, stride)
    # cross_validation_scores(model, X_train, y_train, machine, stories, feature, stride)
    
    evaluate_model(model, model_name, X_test, y_test, machine, stories, feature, stride) 

def main():
    base_path = "models/svm_models/tuned/"
    
    # Walk through all files in the base_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_path = os.path.join(root, file)
                print(f"\nProcessing file: {pkl_path}")
                export_details(pkl_path)
                # print(pkl_path)
                
# main()
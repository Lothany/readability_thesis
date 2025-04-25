import pandas as pd
import numpy as np
import pickle
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
from sklearn.preprocessing import label_binarize
import os
import csv

from random_forest import load_one, load_dataset

def load_model(pkl_path):
    with open(pkl_path, 'rb') as file:
        model = pickle.load(file)
        return model

def parse_dataset(stories, feature, stride):
    model_name =""
    
    if stride == 0:
        print("No stride length filter applied.")
        model_name = f"All N-Grams"
    elif stride == -1:
        print("Returning sentence fragments.")
        model_name = f"Sentence Fragments"
    elif stride in [1, 2, 3, 100]:
        print(f"Filtering dataset with stride length: {stride}")
        model_name = f"N = {stride}"
    else:
        print(f"Invalid stride length: {stride}")
        return
    
    feature_set = ['grade_level', 'text_num', 'stride_len', 'stride_index', 'n_gram']
    if feature == "B":
        feature_set = feature_set
        model_name += " [Trad + Lex]"
    elif feature == "T":
        feature_set += ['noun_tr', 'verb_tr', 'type_tr', 'lex_density', 'lex_foreign']
        model_name += " [Trad]"
    elif feature == "L":
        feature_set += ['sent_len', 'word_len', 'syll_num', 'poly_num']
        model_name += " [Lex]"
    else:
        print("Invalid feature set selected. Please choose 'B', 'T', or 'L'.")
        return
    
    if stories == "full":
        X_train, y_train, X_test, y_test = load_one(stride, feature_set)
    elif stories == "split":
        X_train, y_train, X_test, y_test = load_dataset(stride, feature_set)
    else:
        print("Invalid stories option. Please choose 'full' or 'split'.")
        return

    return X_test, y_test, model_name

def evaluate_model(model, model_name, X_test, y_test, stories, feature, stride):
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
    
    print(f"{stories} - {feature} - {stride}")  
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    

    # Save metrics to CSV
    model_details = {
        "stories": stories,
        "feature": feature,
        "stride": stride,
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1_score": f"{f1:.4f}",
        "roc_auc": f"{roc_auc:.4f}"
    }
    
    performance_scores = "models/rf_records/rf_analysis.csv"
    file_exists = os.path.exists(performance_scores)
    
    with open(performance_scores, "a", newline="", encoding="utf-8") as dataset:
        fieldnames = ["stories", "feature", "stride", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
        writer = csv.DictWriter(dataset, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(model_details)

    model_id = f"{stories}_{feature}_{stride}"
    plot_path = f"models/rf_records/plots"
    
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


def export_details(pkl_path):    
    file_name = pkl_path.split("/")[-1]
    file_name = file_name.replace(".pkl", "")
    parts = file_name.split("_")

    stories = parts[1]
    feature = parts[2]
    stride = int(parts[3])
    
    model = load_model(pkl_path)
    X_test, y_test, model_name = parse_dataset(stories, feature, stride)
    evaluate_model(model, model_name, X_test, y_test, stories, feature, stride) 

def main():
    base_path = "models/rf_records/dataset_full/"
    
    # Walk through all files in the base_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_path = os.path.join(root, file)
                print(f"Processing file: {pkl_path}")
                export_details(pkl_path)
    
main()
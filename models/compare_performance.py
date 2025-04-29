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

import os
import csv
import pickle

from random_forest import load_one, load_dataset

def load_model(pkl_path):
    with open(pkl_path, 'rb') as file:
        model = pickle.load(file)
        return model

def parse_dataset(ml, stories, feature, stride):
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
    
    if stories == "full":
        X_train, y_train, X_test, y_test = load_one(stride, feature_set)
    elif stories == "split":
        X_train, y_train, X_test, y_test = load_dataset(stride, feature_set)
    else:
        print("Invalid stories option. Please choose 'full' or 'split'.")
        return
    
    # Scale to standardize the column values
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train, y_train, X_test, y_test, model_name

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

    model_id = f"{machine}_{stories}_{feature}_{stride}"
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
    _, _, X_test, y_test, model_name = parse_dataset("lr", stories, feature, stride)
    evaluate_model(model, model_name, X_test, y_test, machine, stories, feature, stride) 

def main():
    base_path = "models/lr_models/"
    
    # Walk through all files in the base_path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_path = os.path.join(root, file)
                print(f"\nProcessing file: {pkl_path}")
                export_details(pkl_path)
                # print(pkl_path)
    
# main()
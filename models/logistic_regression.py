import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics


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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

from compare_performance import parse_dataset, model_performance, save_model, evaluate_model
from compare_performance import fold_dataset, split_dataset, export_metrics, export_plot
    
def train_model(X_train, y_train):
    model = LogisticRegression(
        solver='saga',
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model

def tune_model(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        cv=10,
        scoring=['accuracy', 'f1_macro'],  # Use multiple scoring metrics
        refit='accuracy',  # Refit the model using the best accuracy
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)
    
    mean_accuracy = grid.cv_results_['mean_test_accuracy'].max()
    mean_f1_score = grid.cv_results_['mean_test_f1_macro'].max()

    print(f"Cross-Validation Accuracy: {mean_accuracy:.4f}")
    print(f"Cross-Validation F1 Score: {mean_f1_score:.4f}")
    
    return grid.best_estimator_

def create_model(stride, feature, stories, grade):
    machine = "lr"
    
    model_id = f"{machine}_{grade}_{feature}_{stride}"
    print(f"\n\Logistic Regression Model: {model_id}")
    
    train_dataset, test_dataset, feature_set, model_name = parse_dataset(machine, stories, feature, stride, grade)
    
    X_train, y_train, X_test, y_test = split_dataset(train_dataset, test_dataset, feature_set)
       
    # Train logistic regression model
    model = tune_model(X_train, y_train)
    save_model(model, "models/lr_models/", model_id)
    evaluate_model(model, model_name, X_test, y_test, machine, grade, feature, stride)
    
def test():
    stories = "split"
    feature = "B"
    stride = -1
    grade = 1
    
    create_model(stride, feature, "split", grade)
    
def main():
    # stories_list = ["full", "split"]
    stories_list = ["split"]
    features_list = ["B", "T", "L"]
    strides_list = [-1, 0, 1, 2, 3, 100]
    grade_levels = [1, 2, 3, 4, 5, 6]
    
    total_iterations = len(grade_levels) * len(features_list) * len(strides_list)
    
    with tqdm(total=total_iterations, desc="Processing Models", unit="model") as pbar:
        for grade in grade_levels:
            for feature in features_list:
                for stride in strides_list:
                    create_model(stride, feature, "split", grade)
                    pbar.update(1)

main()
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

from compare_performance import parse_dataset, model_performance, save_model, show_cm
    
def train_model(X_train, y_train):
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model

def tune_model(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
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

def test():
    stories = "split"
    feature = "B"
    stride = -1
    grade = 6
    machine = "lr"
    
    model_id = f"TESTING_{machine}_{stories}_{feature}_{stride}"
    print(f"Testing Logistic Regression Model: {model_id}")
    
    X_train, y_train, X_test, y_test, model_name = parse_dataset(machine, stories, feature, stride, grade)
    
    model = train_model(X_train, y_train)
    print(f"Initial Model Performance: ")
    model_performance(model, X_test, y_test)
    show_cm(model, model_name, X_test, y_test, machine, stories, feature, stride)
    
    
    tuned_model = tune_model(X_train, y_train)
    print(f"\nTuned Model Performance: {tuned_model}")
    # model_performance(tuned_model, X_test, y_test)
    
    # create_model(stride, feature, stories)

def create_model(stride, feature, stories):
    model_id = f"lr_{stories}_{feature}_{stride}"
    X_train, y_train, X_test, y_test, model_name  = parse_dataset("lr", stories, feature, stride)
       
    # Train logistic regression model
    model = tune_model(X_train, y_train)
    # save_model(model, "models/lr_models/", model_id)
    model_performance(model, X_test, y_test)
    
def main():
    # stories_list = ["full", "split"]
    stories_list = ["split"]
    features_list = ["B", "T", "L"]
    strides_list = [-1, 0, 1, 2, 3, 100]
    
    total_iterations = len(stories_list) * len(features_list) * len(strides_list)
    
    with tqdm(total=total_iterations, desc="Processing Models", unit="model") as pbar:
        for stories in stories_list:
            for feature in features_list:
                for stride in strides_list:
                    print(f"\nCreating model: {stories}|{feature}|{stride}")
                    create_model(stride, feature, stories)
                    pbar.update(1)

test()
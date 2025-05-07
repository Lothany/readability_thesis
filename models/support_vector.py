# Data Processing
import pandas as pd
import numpy as np
import os  # For path handling

# Modelling
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from tqdm import tqdm
import pickle

from tqdm import tqdm

from compare_performance import parse_dataset, model_performance, save_model
from compare_performance import fold_dataset, split_dataset, export_metrics, export_plot

# --- SVM Specific Training ---

def train_model(X_train, y_train):
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(X_train, y_train):
    param_dist = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale']
    }

    model = SVC(probability=True)
    rand_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        scoring=['accuracy', 'f1_macro'],  # Use multiple scoring metrics
        refit='accuracy',  # Refit the model using the best accuracy
        n_iter=2,
        cv=5,
        random_state=42
    )

    rand_search.fit(X_train, y_train)

    # Extract cross-validation results
    mean_accuracy = rand_search.cv_results_['mean_test_accuracy'].max()
    mean_f1_score = rand_search.cv_results_['mean_test_f1_macro'].max()

    print(f"Cross-Validation Accuracy: {mean_accuracy:.4f}")
    print(f"Cross-Validation F1 Score: {mean_f1_score:.4f}")

    best_model = rand_search.best_estimator_

    print('Best hyperparameters:', rand_search.best_params_)
    
    return best_model

def custom_tune(train_dataset, test_dataset, feature_set):
    param_dist = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['auto']
    }
    
    n_iter=3
    k=5
    
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    best_model = None
    best_accuracy = 0

    for fold in range(k):
        X_train, y_train, X_test, y_test = fold_dataset(train_dataset, test_dataset, feature_set, k, fold)

        base_model = SVC(probability=True)

        # Hyperparameter tuning
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='accuracy',  # or 'f1_macro'
            cv=3,
            refit=True,
            random_state=fold
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        # print('Best hyperparameters:', search.best_params_)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')

        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['roc_auc'].append(roc_auc)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return metrics, best_model

def create_model(stride, feature, stories, grade):
    machine = "svm"
    model_id = f"{machine}_{grade}_{feature}_{stride}"
    print(f"\n\nSupport Vector Machine: {model_id}")
    
    train_dataset, test_dataset, feature_set, model_name = parse_dataset(machine, stories, feature, stride, grade)
    X_train, y_train, X_test, y_test = split_dataset(train_dataset, test_dataset, feature_set)
    
    metrics, model = custom_tune(train_dataset, test_dataset, feature_set)
    
    export_metrics(metrics, machine, grade, feature, stride)
    export_plot(model, model_name, model_id, X_test, y_test)
    save_model(model, "models/svm_models/", model_id)

def test():
    print("Testing SVM Model")
    stories = "split"
    feature = "B"
    stride = 100
    grade = 1
    machine = "svm"
    model_id = f"{machine}_{grade}_{feature}_{stride}"
    print(f"Testing SVM Model: {model_id}")
    
    create_model(stride, feature, "split", grade)

def main():
    # stories_list = ["full", "split"]
    stories_list = ["split"]
    features_list = ["T", "L", "B"]
    strides_list = [-1, 1, 2, 3, 100]
    grade_levels = [3, 4, 5, 6]
    
    total_iterations = len(grade_levels) * len(features_list) * len(strides_list)
    
    with tqdm(total=total_iterations, desc="Processing Models", unit="model") as pbar:
        for grade in grade_levels:
            for feature in features_list:
                for stride in strides_list:
                    create_model(stride, feature, "split", grade)
                    pbar.update(1)

# To run:
main()
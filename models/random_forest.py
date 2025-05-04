import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import display
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

from compare_performance import parse_dataset, model_performance, save_model
from compare_performance import fold_dataset, split_dataset, export_metrics, export_plot
    
# Train the Random Forest Classifier
def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, warm_start=True, random_state=42)
    n_trees = 100

    for i in tqdm(range(1, n_trees + 1), desc="Training"):
        rf.set_params(n_estimators=i)
        rf.fit(X_train, y_train)
    
    return rf

def tune_model(X_train, y_train):
    param_dist = {
    'n_estimators': randint(50, 1000),
    'max_depth': randint(1, 50),
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)

    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_

    print('Best hyperparameters:',  rand_search.best_params_)
    
    return best_rf   

def custom_tune(train_dataset, test_dataset, feature_set):
    k = 5
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    best_model = None
    best_accuracy = 0  # Track the best accuracy to identify the best model

    for fold in range(k):
        X_train, y_train, X_test, y_test = fold_dataset(train_dataset, test_dataset, feature_set, k, fold)

        model = RandomForestClassifier(class_weight='balanced', random_state=fold)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))

        # Update the best model if this fold has the highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return metrics, best_model

def create_model(stride, feature, stories, grade):
    machine = "rf"
    
    model_id = f"{machine}_{grade}_{feature}_{stride}"
    print(f"\n\nRandom Forest Model: {model_id}")
    
    train_dataset, test_dataset, feature_set, model_name = parse_dataset(machine, stories, feature, stride, grade)
    
    X_train, y_train, X_test, y_test = split_dataset(train_dataset, test_dataset, feature_set)
    
    metrics, model = custom_tune(train_dataset, test_dataset, feature_set)
    
    export_metrics(metrics, machine, grade, feature, stride)
    export_plot(model, model_name, model_id, X_test, y_test)
    save_model(model, "models/rf_models/", model_id)
    
    # print("Average metrics across training folds:")
    # for key in metrics:
    #     print(f"{key.capitalize()}: {np.mean(metrics[key]):.4f}")

def test():
    stories = "split"
    feature = "L"
    stride = 1
    grade = 1
    machine = "rf"
    
    model_id = f"{machine}_{grade}_{feature}_{stride}"
    # print(f"Testing Random Forest Model: {model_id}")
    
    train_dataset, test_dataset, feature_set, model_name = parse_dataset(machine, stories, feature, stride, grade)
    # X_train, y_train, X_test, y_test = split_dataset(train_dataset, test_dataset, feature_set)
    
    metrics = custom_tune(train_dataset, test_dataset, feature_set)
    print("Average metrics across training folds:")
    for key in metrics:
        print(f"{key.capitalize()}: {np.mean(metrics[key]):.4f}")

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
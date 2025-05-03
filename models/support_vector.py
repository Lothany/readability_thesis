# Data Processing
import pandas as pd
import numpy as np
import os  # For path handling

# Modelling
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tqdm import tqdm
import pickle

from tqdm import tqdm
from compare_performance import parse_dataset, model_performance, save_model

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

def create_model(stride, feature, stories):
    model_id = f"svm_{stories}_{feature}_{stride}"
    X_train, y_train, _, _, _  = parse_dataset("lr", stories, feature, stride)
    
    model = train_model(X_train, y_train)
    save_model(model, "models/svm_models/untuned/", model_id)
    
    print("\nTuning Model. This may take a while... ")
    tuned_model = hyperparameter_tuning(X_train, y_train)
    save_model(tuned_model, "models/svm_models/tuned/", model_id)

def test():
    print("Testing SVM Model")
    stories = "split"
    feature = "T"
    stride = 100
    model_id = f"svm_{stories}_{feature}_{stride}"
    print(f"Testing SVM Model: {model_id}")
    
    X_train, y_train, X_test, y_test, model_name = parse_dataset("lr", stories, feature, stride)
    
    model = train_model(X_train, y_train)
    # save_model(model, "models/svm_models/untuned/", model_id)
    
    print(f"Initial Model Performance: ")
    model_performance(model, X_test, y_test)
    
    tuned_model = hyperparameter_tuning(X_train, y_train)
    # save_model(tuned_model, "models/svm_models/tuned/", model_id)
    
    print(f"\nTuned Model Performance: ")
    model_performance(tuned_model, X_test, y_test)
    
    # create_model(stride, feature, stories)

def main():
    # stories_list = ["full", "split"]
    stories_list = ["split"]
    features_list = ["B", "L"]
    strides_list = [-1, 0, 1, 2, 3, 100]
    
    total_iterations = len(stories_list) * len(features_list) * len(strides_list)
    
    with tqdm(total=total_iterations, desc="Processing Models", unit="model") as pbar:
        for stories in stories_list:
            for feature in features_list:
                for stride in strides_list:
                    print(f"Creating model for {feature} features, and stride length {stride} using {stories} stories.")
                    create_model(stride, feature, stories)
                    pbar.update(1)

# To run:
test()
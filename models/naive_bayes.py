import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

from tqdm import tqdm
from random_forest import save_model
from compare_performance import parse_dataset

def model_performance(model, X_test, y_test):
    # y_pred = model.predict(X_test)    
    # score = accuracy_score(y_test, y_pred)
    # return score
    
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
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

def train_model(X_train, y_train):
    # Use GaussianNB for continuous data or MultinomialNB for discrete data
    model = GaussianNB()  # Replace with MultinomialNB() if your data is discrete
    model.fit(X_train, y_train)
    return model

def tune_model(X_train, y_train):
    # Naive Bayes models typically have fewer hyperparameters to tune.
    # For GaussianNB, you might tune `var_smoothing`.
    param_grid = {
        'var_smoothing': np.logspace(-9, 0, 10)  # Example for GaussianNB
    }

    grid = GridSearchCV(
        estimator=GaussianNB(),
        param_grid=param_grid,
        cv=5,
        scoring=['accuracy', 'f1_macro'],  # Use multiple scoring metrics
        refit='accuracy',  # Refit the model using the best accuracy
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    # Extract cross-validation results
    mean_accuracy = grid.cv_results_['mean_test_accuracy'].max()
    mean_f1_score = grid.cv_results_['mean_test_f1_macro'].max()

    print(f"Cross-Validation Accuracy: {mean_accuracy:.4f}")
    print(f"Cross-Validation F1 Score: {mean_f1_score:.4f}")

    return grid.best_estimator_

def test():
    stories = "split"
    feature = "T"
    stride = 100
    # model_id = f"lr_{stories}_{feature}_{stride}"
    
    # X_train, y_train, X_test, y_test, model_name = parse_dataset("lr", stories, feature, stride)
    
    # model = train_model(X_train, y_train)
    # print(f"Initial Model Performance: ")
    # model_performance(model, X_test, y_test)
    
    
    # tuned_model = tune_model(X_train, y_train)
    # print(f"\nTuned Model Performance: ")
    # model_performance(tuned_model, X_test, y_test)
    
    create_model(stride, feature, stories)

def create_model(stride, feature, stories):
    model_id = f"lr_{stories}_{feature}_{stride}"
    X_train, y_train, X_test, y_test, model_name = parse_dataset("lr", stories, feature, stride)
       
    # Train logistic regression model
    model = train_model(X_train, y_train)
    # save_model(model, "models/lr_models/", model_id)
    print(f"Initial Model Performance: ")
    model_performance(model, X_test, y_test)
    
    
    tuned_model = tune_model(X_train, y_train)
    print(f"\nTuned Model Performance: {tuned_model}")
    # model_performance(tuned_model, X_test, y_test)
    
    
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

# test()
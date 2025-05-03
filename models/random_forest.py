import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import display
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

from compare_performance import parse_dataset, model_performance, save_model, show_cm
    
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
    
def create_model(stride, feature, stories):
    model_id = f"rf_{stories}_{feature}_{stride}"
    X_train, y_train, _, _, _  = parse_dataset("lr", stories, feature, stride)
    
    model = train_model(X_train, y_train)
    save_model("rf", model, "models/random_forest/", model_id)
    
    print("\nTuning Model. This may take a while... ")
    tuned_model = tune_model(X_train, y_train)
    save_model("rf", tuned_model, "models/random_forest/tuned_", model_id)

def test():
    stories = "split"
    feature = "B"
    stride = -1
    grade = 6
    machine = "rf"
    
    model_id = f"rf_{stories}_{feature}_{stride}"
    print(f"Testing Logistic Regression Model: {model_id}")
    
    X_train, y_train, X_test, y_test, model_name = parse_dataset(machine, stories, feature, stride, grade)
    print(X_train.head())
    
    model = train_model(X_train, y_train)
    print(f"Initial Model Performance: ")
    model_performance(model, X_test, y_test)
    show_cm(model, model_name, X_test, y_test, machine, stories, feature, stride)
    
    
    # tuned_model = tune_model(X_train, y_train)
    # print(f"\nTuned Model Performance: {tuned_model}")
    # model_performance(tuned_model, X_test, y_test)
    
    # create_model(stride, feature, stories)

def main():
    stories_list = ["full", "split"]
    features_list = ["B", "T", "L"]
    strides_list = [-1, 0, 1, 2, 3, 100]
    
    total_iterations = len(stories_list) * len(features_list) * len(strides_list)
    
    with tqdm(total=total_iterations, desc="Processing Models", unit="model") as pbar:
        for stories in stories_list:
            for feature in features_list:
                for stride in strides_list:
                    print(f"Creating model for {feature} features, and stride length {stride} using {stories} stories.")
                    create_model(stride, feature, stories)
                    pbar.update(1)

test()
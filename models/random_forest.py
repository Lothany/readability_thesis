# Data Processing
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import display
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns


# Modules
from compare_performance import parse_dataset, print_scores
    
# Train the Random Forest Classifier
def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, warm_start=True, random_state=42)
    n_trees = 100

    for i in tqdm(range(1, n_trees + 1), desc="Training"):
        rf.set_params(n_estimators=i)
        rf.fit(X_train, y_train)
    
    return rf

def hyperparameter_tuning(X_train, y_train):
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
    
def save_model(model, root_path, model_name):
    pkl_path = f"{root_path}{model_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    
        
def create_model(stride, feature, grade):
    model_id = f"{grade}_{feature}_{stride}"
    X_train, y_train, _, _, _ = parse_dataset("rf", grade, feature, stride)
    
    # model = train_model(X_train, y_train)
    # save_model("rf", model, "models/random_forest/", model_id)
    
    print("\nTuning Model. This may take a while... ")
    tuned_model = hyperparameter_tuning(X_train, y_train)
    save_model("rf", tuned_model, "models/random_forest/", model_id)

def test():
    grade = 3
    feature = "B"
    stride =  3
    model_id = f"rf_{grade}_{feature}_{stride}"
    print(f"Model: {model_id}")
    
    X_train, y_train, X_test, y_test, model_name = parse_dataset("rf", grade, feature, stride)
    
    model = train_model(X_train, y_train)
    # save_model(model, "models/svm_models/untuned/", model_id)
    
    print(f"Initial Model Performance: ")
    print_scores(model, X_test, y_test)
    
    tuned_model = hyperparameter_tuning(X_train, y_train)
    # save_model(tuned_model, "models/svm_models/tuned/", model_id)
    
    print(f"\nTuned Model Performance: ")
    print_scores(tuned_model, X_test, y_test)

def main():
    grades_list = [1, 2, 3, 4, 5, 6]
    features_list = ["B"]
    strides_list = [-1, 100]
    
    total_iterations = len(grades_list) * len(features_list) * len(strides_list)
    
    with tqdm(total=total_iterations, desc="Processing Models", unit="model") as pbar:
        for grade in grades_list:
            for feature in features_list:
                for stride in strides_list:
                    print(f"Creating Grade {grade} model for {feature} features, and stride length {stride}.")
                    create_model(stride, feature, grade)
                    pbar.update(1)

test()
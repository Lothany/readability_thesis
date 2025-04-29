import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics


from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from random_forest import load_one, load_dataset, save_model



def parse_dataset(type, stories, feature, stride):
    model_name =f"{type}"
    
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

    return X_train, y_train, X_test, y_test, model_name

def model_performance(model, X_test, y_test):
    score = model.score(X_test, y_test)
    return score
    

def train_model(X_train, y_train):
    # Use OneVsRestClassifier to handle multi-class classification
    lm = OneVsRestClassifier(linear_model.LogisticRegression(solver='liblinear'))
    lm.fit(X_train, y_train)
    
    return lm

def create_model(stride, feature, stories):
    model_id = f"lr_{stories}_{feature}_{stride}"
    X_train, y_train, X_test, y_test, model_name = parse_dataset("lr", stories, feature, stride)
    
    
    
    # Train logistic regression model
    model = train_model(X_train, y_train)
    score = model_performance(model, X_test, y_test)

    
    
    print(f"{model_id}: {score}")

# def tune_model(model):
    
    

def test():
    stories = "full"
    feature = "L"
    stride = -1
    model_id = f"lr_{stories}_{feature}_{stride}"
    
    # Load the dataset acc to the parameters (stride, feature, stories)
    # Get X and y for training and testing
    X_train, y_train, X_test, y_test, model_name = parse_dataset("lr", stories, feature, stride)
    
    # Train logistic regression model
    model = train_model(X_train, y_train)
    score = model_performance(model, X_test, y_test)
    
    

def test2():
    stories_list = ["full", "split"]
    stories_list = ["full"]
    features_list = ["B", "T", "L"]
    strides_list = [-1, 0, 1, 2, 3, 100]
    
    total_iterations = len(stories_list) * len(features_list) * len(strides_list)
    
    with tqdm(total=total_iterations, desc="Processing Models", unit="model") as pbar:
        for stories in stories_list:
            for feature in features_list:
                for stride in strides_list:
                    print(f"\nCreating model for {feature} features, and stride length {stride} using {stories} stories.")
                    create_model(stride, feature, stories)
                    pbar.update(1)

test()
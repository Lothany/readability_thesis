# Data Processing
import pandas as pd
import numpy as np
import os  # For path handling

# Modelling
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tqdm import tqdm
import pickle

# --- Dataset Processing Functions ---

def filter_dataset(dataset, stride_length):
    if stride_length == -1:
        filtered_dataset = dataset[(dataset['stride_len'] > 3) & (dataset['stride_len'] < 100)]
    elif stride_length == 0:
        return dataset
    else:
        filtered_dataset = dataset[dataset['stride_len'] == stride_length]
    return filtered_dataset

def load_one(stride_length, feature_set):
    dataset_source = 'tables/allbooks_dataset.csv'
    
    df = pd.read_csv(dataset_source)
    df = filter_dataset(df, stride_length)
    df = df[df['text_num'] != 18]
    df = df.drop(columns=['word_num'])
    
    X = df.drop(columns=feature_set)
    y = df['grade_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test

def load_dataset(stride_length, feature_set):
    training_source = 'tables/dataset.csv'
    testing_source = 'tables/dataset_testing.csv'
    
    train_dataset = pd.read_csv(training_source)
    train_dataset = filter_dataset(train_dataset, stride_length)
    train_dataset = train_dataset[train_dataset['text_num'] != 18]
    train_dataset = train_dataset.drop(columns=['word_num'])
    
    X_train = train_dataset.drop(columns=feature_set)
    y_train = train_dataset['grade_level']

    test_dataset = pd.read_csv(testing_source)
    test_dataset = filter_dataset(test_dataset, stride_length)
    test_dataset = test_dataset.drop(columns=['word_num'])

    X_test = test_dataset.drop(columns=feature_set)
    y_test = test_dataset['grade_level']
    
    return X_train, y_train, X_test, y_test

def save_model(model, root_path, model_name):
    # Ensure the directory exists
    os.makedirs(root_path, exist_ok=True)  # Create the directory if it doesn't exist
    pkl_path = os.path.join(root_path, f"{model_name}.pkl")  # Improved path handling
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)

# --- SVM Specific Training ---

def train_model(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(X_train, y_train):
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    model = SVC(probability=True)
    rand_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=2,
        cv=2,
        random_state=42
    )

    rand_search.fit(X_train, y_train)
    best_model = rand_search.best_estimator_

    print('Best hyperparameters:', rand_search.best_params_)
    
    return best_model

def create_model(stride, feature, stories):
    model_id = f"{stories}_{feature}_{stride}"
    
    if stride == 0:
        print("No stride length filter applied.")
        model_name = f"All N-Grams"
    elif stride == -1:
        print("Returning sentence fragments.")
        model_name = f"Sentence Fragments"
    elif stride in [1, 2, 3, 100]:
        print(f"Filtering dataset with stride length: {stride}")
        model_name = f"N = {stride}"
    else:
        print(f"Invalid stride length: {stride}")
        return
    
    feature_set = ['grade_level', 'text_num', 'stride_len', 'stride_index', 'n_gram']
    if feature == "B":
        feature_set = feature_set
    elif feature == "T":
        feature_set += ['noun_tr', 'verb_tr', 'type_tr', 'lex_density', 'lex_foreign']
    elif feature == "L":
        feature_set += ['sent_len', 'word_len', 'syll_num', 'poly_num']
    else:
        raise ValueError("Invalid feature set selected. Please choose 'B', 'T', or 'L'.")  # Changed to raise an exception
    
    if stories == "full":
        X_train, y_train, X_test, y_test = load_one(stride, feature_set)
    elif stories == "split":
        X_train, y_train, X_test, y_test = load_dataset(stride, feature_set)
    else:
        raise ValueError("Invalid stories option. Please choose 'full' or 'split'.")  # Changed to raise an exception
    
    model = train_model(X_train, y_train, X_test, y_test)
    save_model(model, "models/svm/", model_id)
    
    print("\nTuning Model. This may take a while... ")
    tuned_model = hyperparameter_tuning(X_train, y_train)
    save_model(tuned_model, "models/svm/tuned_", model_id)

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

# To run:
main()  # Uncommented to allow execution
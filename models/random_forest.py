# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import display
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns


# Progress Bar
from tqdm import tqdm

# Filter dataset based on stride length
def filter_dataset(dataset, stride_length):
    # Return sentence fragments
    if stride_length == -1:
        filtered_dataset = dataset[(dataset['stride_len'] > 3) & (dataset['stride_len'] < 100)]
    # Do not filter
    elif stride_length == 0:
        return dataset
    else:
        filtered_dataset = dataset[dataset['stride_len'] == stride_length]
    
    return filtered_dataset

def load_dataset(training_source, testing_source, stride_length):
    train_dataset= pd.read_csv(training_source)

    # Filter out rows with from 18.txt and drop empty column word_len
    train_dataset = train_dataset[train_dataset['text_num'] != 18]
    train_dataset = train_dataset.drop(columns=['word_num'])

    # Execute for specific stride length
    train_dataset = filter_dataset(train_dataset, stride_length)

    # dataset['grade_level'] = dataset['grade_level'].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6})
    # print(train_dataset)

    # Split the dataset into features and target variable
    X_train = train_dataset.drop(columns=['grade_level', 'text_num', 'stride_len', 'stride_index', 'n_gram'])
    y_train = train_dataset['grade_level']

    # Load testing dataset
    test_dataset = pd.read_csv(testing_source)
    test_dataset = test_dataset.drop(columns=['word_num'])
    test_dataset = filter_dataset(test_dataset, stride_length)

    # test_dataset['grade_level'] = test_dataset['grade_level'].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6})
    # print(test_dataset)

    X_test = test_dataset.drop(columns=['grade_level', 'text_num', 'stride_len', 'stride_index', 'n_gram'])
    y_test = test_dataset['grade_level']
    
    return X_train, y_train, X_test, y_test
    
# Train the Random Forest Classifier
def train_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=42)
    n_trees = 100

    for i in tqdm(range(1, n_trees + 1), desc="Training Progress"):
        rf.set_params(n_estimators=i)
        rf.fit(X_train, y_train)
    
    return rf
    
def evaluate_model(X_train, y_train, X_test, y_test, model):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}\n")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n {cm}\n")
    
    feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(f"Feature Importance:\n{feature_scores}\n")
    
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
    

def feature_importance(X_train, y_train, X_test, y_test, model):
    feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")

    plt.show()
    
def model_accuracy():
    training_source = 'tables/dataset.csv'
    testing_source = 'tables/dataset_testing.csv'
    
    for stride in [-1, 0, 1, 2, 3, 100]:    
        X_train, y_train, X_test, y_test = load_dataset(training_source, testing_source, stride)
        model = train_model(X_train, y_train, X_test, y_test)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if stride == 0:
            print("No stride length filter applied.")
        elif stride == -1:
            print("Sentence fragments.")
        else:
            print(f"Stride length: {stride}")
        
        print(f"Accuracy Score: {accuracy}\n")
        
    
def main():
    # Input desired n-gram length [1, 2, 3, 100]
    # Enter 0 to skip filter and -1 to return sentence fragments
    stride_length = 100
    
    if stride_length == 0:
        print("No stride length filter applied.")
    elif stride_length == -1:
        print("Returning sentence fragments.")
    elif stride_length in [1, 2, 3, 100]:
        print(f"Filtering dataset with stride length: {stride_length}")
    else:
        print(f"Invalid stride length: {stride_length}")
        return
    
    training_source = 'tables/dataset.csv'
    testing_source = 'tables/dataset_testing.csv'
    
    X_train, y_train, X_test, y_test = load_dataset(training_source, testing_source, stride_length)
    model = train_model(X_train, y_train, X_test, y_test)
    # evaluate_model(X_train, y_train, X_test, y_test, model)
    # feature_importance(X_train, y_train, X_test, y_test, model)
    
model_accuracy()
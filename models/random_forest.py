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

def load_one(dataset_source, stride_length):
    df= pd.read_csv(dataset_source)
    df = filter_dataset(df, stride_length)
    
    df = df[df['text_num'] != 18]
    df = df.drop(columns=['word_num'])
    
    X = df.drop(columns=['grade_level', 'text_num', 'stride_len', 'stride_index', 'n_gram'])
    y = df['grade_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, y_train, X_test, y_test
    
def load_dataset(training_source, testing_source, stride_length):
    train_dataset= pd.read_csv(training_source)
    train_dataset = filter_dataset(train_dataset, stride_length)
    # train_dataset = undersample_dataset(train_dataset)
    
    # Filter out rows with from 18.txt and drop empty column word_len
    train_dataset = train_dataset[train_dataset['text_num'] != 18]
    
    # train_dataset = train_dataset[train_dataset['grade_level'] != 1]
    # train_dataset = train_dataset[train_dataset['grade_level'] != 2]
    # train_dataset = train_dataset[train_dataset['grade_level'] != 4]
    
    train_dataset = train_dataset.drop(columns=['word_num'])

    # dataset['grade_level'] = dataset['grade_level'].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6})
    # print(train_dataset)

    # Split the dataset into features and target variable
    X_train = train_dataset.drop(columns=['grade_level', 'text_num', 'stride_len', 'stride_index', 'n_gram'])
    y_train = train_dataset['grade_level']

    # Load testing dataset
    test_dataset = pd.read_csv(testing_source)
    test_dataset = filter_dataset(test_dataset, stride_length)
    
    # test_dataset = test_dataset[test_dataset['grade_level'] != 1]
    # test_dataset = test_dataset[test_dataset['grade_level'] != 2]
    # test_dataset = test_dataset[test_dataset['grade_level'] != 4]
    
    test_dataset = test_dataset.drop(columns=['word_num'])

    # test_dataset['grade_level'] = test_dataset['grade_level'].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6})
    # print(test_dataset)

    X_test = test_dataset.drop(columns=['grade_level', 'text_num', 'stride_len', 'stride_index', 'n_gram'])
    y_test = test_dataset['grade_level']
    
    return X_train, y_train, X_test, y_test

def undersample_dataset(df):
    # Find the minimum count of entries across all grade levels
    min_count = df['grade_level'].value_counts().min()
    
    # Group by grade level and sample min_count entries from each group
    undersampled_df = df.groupby('grade_level').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    return undersampled_df
    
# Train the Random Forest Classifier
def train_model(X_train, y_train, X_test, y_test):
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
    
def evaluate_model(X_train, y_train, X_test, y_test, model):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}\n")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n {cm}\n")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')  # Customize the colormap and format
    plt.title("Confusion Matrix")
    plt.show()
    
    
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
        # X_train, y_train, X_test, y_test = load_dataset(training_source, testing_source, stride)
        X_train, y_train, X_test, y_test = load_one(training_source, stride
                                                    )
        model = train_model(X_train, y_train, X_test, y_test)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if stride == 0:
            print("No stride length filter applied.")
        elif stride == -1:
            print("Sentence fragments.")
        else:
            print(f"Stride length: {stride}")
        
        # feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        # print(f"Feature Importance:\n{feature_scores}\n")
        print(f"Accuracy Score: {accuracy}\n")
        
    
def main():
    # Input desired n-gram length [1, 2, 3, 100]
    # Enter 0 to skip filter and -1 to return sentence fragments
    stride_length = 1
    
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
    
    # X_train, y_train, X_test, y_test = load_dataset(training_source, testing_source, stride_length)
    
    X_train, y_train, X_test, y_test = load_one(training_source, stride_length)
    
    print("\nInitial Model")
    model = train_model(X_train, y_train, X_test, y_test)
    evaluate_model(X_train, y_train, X_test, y_test, model)
    # feature_importance(X_train, y_train, X_test, y_test, model)
    
    print("\nTuned Model")
    tuned_model = hyperparameter_tuning(X_train, y_train)
    evaluate_model(X_train, y_train, X_test, y_test, tuned_model)
    
main()
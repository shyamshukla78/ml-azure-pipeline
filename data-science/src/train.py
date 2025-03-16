# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using the training dataset and evaluates using the test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--criterion', type=str, default='gini',
                        help='The function to measure the quality of a split')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from CSV
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Split the data into input(X) and output(y)
    y_train = train_df['Failure']
    X_train = train_df.drop(columns=['Failure'])
    y_test = test_df['Failure']
    X_test = test_df.drop(columns=['Failure'])

    # Initialize and train a Decision Tree Classifier
    model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # Log model hyperparameters
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict using the Decision Tree Model on test data
    yhat_test = model.predict(X_test)

    # Compute and log accuracy score
    accuracy = accuracy_score(y_test, yhat_test)
    print(f'Accuracy of Decision Tree classifier on test set: {accuracy:.2f}')
    # Logging the accuracy score as a metric
    mlflow.log_metric("Accuracy", float(accuracy))

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Criterion: {args.criterion}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

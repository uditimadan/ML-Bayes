from collections import defaultdict
from random import shuffle

import classifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class my_naive_bayes():

    def __init__(self, filename):
        self.features = None
        self.classifications = None
        self.filename = "breast-cancer.data"
        self.pos_examples = defaultdict(dict)  # Nested defaultdict for attributes and values
        self.neg_examples = defaultdict(dict)  # Nested defaultdict for attributes and values

        # Load data (assuming you still need it)
        self.data = pd.read_csv(self.filename, names=["Class", 'age', 'menopause', 'tumor-size',
                                          'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                                          'breast-quad', 'irradiated'])

    def update_examples(self, classification, example):
        for j, attr in enumerate(self.data.columns[1:]):
            example_value = example[j]
            if classification == 'no-recurrence':
                self.neg_examples[attr][example_value] = self.neg_examples[attr].get(example_value, 0) + 1
            else:
                self.pos_examples[attr][example_value] = self.pos_examples[attr].get(example_value, 0) + 1

    def fit(self, examples, classifications):
        for i, classification in enumerate(classifications):
            example = examples[i]
            self.update_examples(classification, example)

    def predict(self, examples):
        predictions = []
        for example in examples:
            pos_log_likelihood = 0
            neg_log_likelihood = 0
            threshold = 3  # Example threshold for early stopping (adjustable)
            for j, attr in enumerate(self.data.columns[1:]):
                pos_count = self.pos_examples[attr].get(example[j], 0) + 1  # Laplace smoothing
                neg_count = self.neg_examples[attr].get(example[j], 0) + 1  # Laplace smoothing
                pos_log_likelihood += np.log(pos_count)
                neg_log_likelihood += np.log(neg_count)

                # Early stopping if difference is significant
                if abs(pos_log_likelihood - neg_log_likelihood) > threshold:
                    break

            # Add class prior probabilities (log)
            pos_log_likelihood += np.log(len(self.pos_examples) / (len(self.pos_examples) + len(self.neg_examples)))
            neg_log_likelihood += np.log(len(self.neg_examples) / (len(self.pos_examples) + len(self.neg_examples)))

            if pos_log_likelihood > neg_log_likelihood:
                predictions.append('recurrence')
            else:
                predictions.append('no-recurrence')
        return predictions

    def score(self, predicted, actual):
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average='binary', pos_label='recurrence')
        return f1

    # Function to implement five-fold cross-validation

    def five_fold(self):
        # Load data
        data = pd.read_csv("breast-cancer.data", names=["Class", 'age', 'menopause', 'tumor-size',
                                                        'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                                                        'breast-quad', 'irradiated'])

        # Shuffle data
        data = shuffle(data)

        # Perform 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True)  # Ensure shuffling within folds
        f1_scores = []

        for train_index, test_index in kf.split(data):
            # Split data into training and testing sets using indexing
            train_examples = data.iloc[train_index][data.columns[1:]]
            train_classifications = data["Class"].iloc[train_index]

            # Check for consistent sample size before fitting (optional)
            if len(train_examples) != len(train_classifications):
                raise ValueError("Inconsistent sample sizes in training data!")

            test_examples = data.iloc[test_index][data.columns[1:]]
            test_classifications = data["Class"].iloc[test_index]

            # Create new classifier, fit on training data, and predict on testing data
            nb_classifier = my_naive_bayes("")
            nb_classifier.fit(train_examples, train_classifications)
            predicted_results = nb_classifier.predict(test_examples)

            # Ensure consistent sample size before scoring (optional)
            if len(predicted_results) != len(test_classifications):
                raise ValueError("Inconsistent sample sizes in prediction results!")

            # Calculate and store F1 score for the fold
            f1_score_fold = nb_classifier.score(predicted_results, test_classifications)
            f1_scores.append(f1_score_fold)

        return f1_scores


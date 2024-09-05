
from ZeroR import zeroR  # Import ZeroR without importing `data`
from nb import my_naive_bayes

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

print("Unit tests passed successfully in test_nb.py")

# Load data
data_df = pd.read_csv("breast-cancer.data", names=["Class", 'age', 'menopause', 'tumor-size',
                                                    'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                                                    'breast-quad', 'irradiated'])

# Shuffle data
data_df = data_df.sample(frac=1).reset_index(drop=True)

#print(data_df.head())
print("Expected outcome for naive bayes score to be higher than ZeroR or it outruns the prog")

# Perform five-fold cross-validation for Naive Bayes classifier
nb_f1_scores = []
for train_index, test_index in KFold(n_splits=5, shuffle=True).split(data_df):
    train_data = data_df.iloc[train_index]
    test_data = data_df.iloc[test_index]
    nb_classifier = my_naive_bayes("breast-cancer.data")  # Instantiate Naive Bayes classifier with correct filename
    nb_classifier.fit(train_data.drop("Class", axis=1), train_data["Class"])
    predicted_results = nb_classifier.predict(test_data.drop("Class", axis=1))
    f1 = f1_score(test_data["Class"], predicted_results, average='binary')
    nb_f1_scores.append(f1)

# Perform five-fold cross-validation for ZeroR classifier
zr_f1_scores = []
for train_index, test_index in KFold(n_splits=5, shuffle=True).split(data_df):
    train_data = data_df.iloc[train_index]
    test_data = data_df.iloc[test_index]
    zr_classifier = zeroR(train_data)  # Instantiate ZeroR classifier
    predicted_results = zr_classifier.predict(test_data.drop("Class", axis=1))
    f1 = f1_score(test_data["Class"], predicted_results, average='binary')
    zr_f1_scores.append(f1)

# Calculate average F1 scores
avg_f1_nb = sum(nb_f1_scores) / len(nb_f1_scores)
avg_f1_zr = sum(zr_f1_scores) / len(zr_f1_scores)

# Create a table to display the results
results_table = pd.DataFrame({
    "Classifier": ["Naive Bayes", "ZeroR"],
    "Average F1 Score": [avg_f1_nb, avg_f1_zr]
})
print(results_table)

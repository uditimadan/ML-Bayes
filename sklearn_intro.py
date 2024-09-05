
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

# loading the wine dataset
wine_data = load_wine()
features = wine_data.data
classification = wine_data.target

# Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(features, classification)

results = gnb.predict(features)
# the listed tuple results from the wine dataset
#print(list(zip(results, classification)))


# 5-fold cross-validation score
print('Cross-validation F1 score:',gnb.score(features, classification))
#print(cross_val_score(gnb))

# 5-fold cross-validation
# Updated with F1 score for evaluating performance
f1_score = cross_val_score(gnb, features, classification, cv=5, scoring='f1_micro')
print('Evaluated cross-validation F1 scores:', f1_score)





import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

# Reading the CSV into dataframe
df = pd.read_csv("../pd_speech_features.csv", header=1)
df.drop("id", inplace=True, axis=1)

# Data Preprocessing
X = df.iloc[:, 0:753].values
y = df.iloc[:, 753].values

# Splitting Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Fitting the RandomForest Model
clf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Displaying Accuracy, F1 Score and R^2 values
print("F1-Score: {}".format(f1_score(y_test, y_pred)))
print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
print("R^2 Value: {}".format(r2_score(y_test, y_pred)))

# No Skill Prediction
ns_probs = [0 for _ in range(len(y_test))]

# Random Forest AUC
rf_auc = roc_auc_score(y_test, y_pred)
print("AUC: {}".format(rf_auc))

# ROC Curve
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
pyplot.plot(rf_fpr, rf_tpr, marker=".", label="Random Forest")
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.legend()
pyplot.show()

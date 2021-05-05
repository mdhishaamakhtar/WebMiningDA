import pandas as pd
import pydotplus
from IPython.display import Image
from matplotlib import pyplot
from six import StringIO
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Reading the CSV into dataframe
df = pd.read_csv("../pd_speech_features.csv", header=1)

# Data Preprocessing
df.drop("id", inplace=True, axis=1)
X = df.iloc[:, 0:753].values
y = df.iloc[:, 753].values

# Splitting Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Fitting the DecisionTree Model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=100, random_state=7)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Displaying Accuracy, F1 Score and R^2 values
print("F1-Score: {}".format(f1_score(y_test, y_pred)))
print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
print("R^2 Value: {}".format(r2_score(y_test, y_pred)))

# No Skill Prediction
ns_probs = [0 for _ in range(len(y_test))]

# Decision Tree AUC
dt_auc = roc_auc_score(y_test, y_pred)
print("AUC: {}".format(dt_auc))

# ROC Curve
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, y_pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
pyplot.plot(dt_fpr, dt_tpr, marker=".", label="Decision Trees")
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.legend()
pyplot.show()

# Exporting constructed decision tree graph as an image
dot_data = StringIO()
export_graphviz(
    clf,
    out_file=dot_data,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=df.iloc[:, :753].columns,
    class_names=["0", "1"],
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("DecisionTreeGraph.png")
Image(graph.create_png())

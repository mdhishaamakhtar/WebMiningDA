# Web Mining Digital Assignment

#### Course Code: CSE3024

### Faculty

- Dr. Shashank Mouli Satapathy

### Members

- Md Hishaam Akhtar 19BCE0300
- Jeswin Jacob J 18BCE2121

### Contents:
- Datasets
    - [Sales Transactions Dataset Weekly](Sales_Transactions_Dataset_Weekly.csv)
    - [PD Speech Features](pd_speech_features.csv)

- [Classification](Classification)
    - Random Forest
      - [Random Forest Implementation](Classification/RandomForest.py)
      - [Random Forest ROC Curve](Classification/RandomForestROC.png)
      - [Random Forest Performance](Classification/RandomForestPerformance.png)
    - Decision Tree
      - [Decision Tree Implementation](Classification/DecisionTree.py)
      - [Decision Tree ROC Curve](Classification/DecisionTreeROC.png)
      - [Decision Tree Performance](Classification/DecisionTreePerformance.png)
      - [Decision Tree Graph](Classification/DecisionTreeGraph.png)

- [Clustering](Clustering)
    - KMeans Clustering
      - [KMeans Clustering Implementation](Clustering/KMeans.py)
      - [KMeans Elbow](Clustering/KMeansElbow.png)
      - [KMeans Scatter Plot](Clustering/KMeansScatter.png)
    - Probability Based Clustering
      - [Probability Based Clustering Implementation](Clustering/ProbabilityBased.py)
      - [Probability Based Clustering Scatter](Clustering/ProbabilityBasedScatter.png)

### Instructions to Run:

- Creating virtual environment

```bash
pip install virtualenv
virtualenv env
env\Scripts\activate # Windows
source env/bin/activate # Linux and MacOS
```

- Installing required packages

```bash
pip install -r requirements.txt
```

- Classification
    - Random Forest
    ```bash
    python RandomForest.py
    ```
    - Decision Tree
    ```bash
    python DecisionTree.py
    ```
- Clustering
    - KMeans Clustering
    ```bash
    python KMeans.py
    ```
    - Probability Based
    ```bash
    python ProbabilityBased.py
    ```
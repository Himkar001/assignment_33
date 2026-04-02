# Assignment 33 AM + PM Session

## Overview

This assignment focuses on applying core machine learning concepts across multiple problem types including classification, similarity search, and model selection. The AM session emphasizes practical implementation using real datasets such as handwritten digits and explores performance optimization using FAISS. The PM session focuses more on conceptual understanding, model comparison, and applying machine learning techniques to text data.

The assignment demonstrates the ability to:

* Train and evaluate multiple ML models
* Perform hyperparameter tuning
* Compare models using appropriate metrics
* Implement algorithms from scratch
* Understand performance trade-offs and optimization techniques

---

# AM Session

---

## Part A: Handwritten Digit Classification (SVM vs KNN)

### Objective

The goal of this part is to build a classifier using Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) on the sklearn digits dataset and compare their performance using accuracy, confusion matrix, and classification report.

### Explanation

The digits dataset consists of 8x8 grayscale images flattened into 64 features. Since both SVM and KNN rely on distance-based calculations, feature scaling is necessary. SVM with RBF kernel is used for capturing non-linear patterns, while KNN classifies based on nearest neighbors.

### Code

```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}

svm = SVC(kernel='rbf')

grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_svm = grid.best_estimator_
```

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svm_preds = best_svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_preds)
svm_cm = confusion_matrix(y_test, svm_preds)
svm_report = classification_report(y_test, svm_preds)

svm_acc, svm_cm, svm_report
```

```python
from sklearn.neighbors import KNeighborsClassifier

k_values = [1, 3, 5, 7, 9]
knn_scores = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    knn_scores[k] = accuracy_score(y_test, preds)

best_k = max(knn_scores, key=knn_scores.get)

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

best_k, knn_scores
```

```python
knn_preds = best_knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_preds)
knn_cm = confusion_matrix(y_test, knn_preds)
knn_report = classification_report(y_test, knn_preds)

knn_acc, knn_cm, knn_report
```

```python
cm = svm_cm.copy()
np.fill_diagonal(cm, 0)

confusions = []

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i][j] > 0:
            confusions.append(((i, j), cm[i][j]))

sorted_confusions = sorted(confusions, key=lambda x: x[1], reverse=True)

sorted_confusions[:10]
```

### Results

* SVM Accuracy ≈ 0.98
* KNN Accuracy ≈ 0.97
* Most confused digits: (3,8), (4,9), (1,7)

### Conclusion

SVM performs better due to its ability to learn complex boundaries. KNN performs well but is slightly less accurate and slower during prediction.

---

## Part B: Approximate Nearest Neighbors using FAISS

### Objective

To compare traditional KNN with FAISS for fast nearest neighbor search and evaluate performance differences.

### Explanation

KNN becomes slow when dataset size increases because it computes distance for all points. FAISS is optimized for fast similarity search using vector indexing and is widely used in large-scale systems.

### Code

```python
!pip install faiss-cpu
```

```python
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
```

```python
from sklearn.neighbors import KNeighborsClassifier
import time

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

start = time.time()
knn_preds = knn.predict(X[:1000])
knn_time = time.time() - start

knn_time
```

```python
import faiss
import numpy as np

X_np = np.array(X).astype('float32')

index = faiss.IndexFlatL2(X_np.shape[1])
index.add(X_np)
```

```python
queries = X_np[:1000]

start = time.time()
distances, indices = index.search(queries, 3)
faiss_time = time.time() - start

faiss_time
```

```python
knn_time, faiss_time
```

### Results

* sklearn KNN Time ≈ 0.02–0.05 sec
* FAISS Time ≈ 0.005–0.01 sec

### Conclusion

FAISS significantly improves speed and is suitable for large datasets where traditional KNN becomes inefficient.

---

## Part C: Interview Ready

### Q1: Conceptual

Logistic Regression focuses on minimizing log loss and predicting probabilities, while SVM focuses on maximizing margin between classes using support vectors. Logistic Regression is preferred when interpretability and probability outputs are needed. SVM is better suited for high-dimensional data and complex boundaries.

---

### Q2: Coding

```python
import numpy as np

def knn_from_scratch(X_train, y_train, X_test, k):

    predictions = []

    for test_point in X_test:

        distances = np.sqrt(((X_train - test_point) ** 2).sum(axis=1))

        k_indices = np.argsort(distances)[:k]

        k_labels = y_train[k_indices]

        values, counts = np.unique(k_labels, return_counts=True)

        pred = values[np.argmax(counts)]

        predictions.append(pred)

    return np.array(predictions)
```

---

### Q3: Debug

The issue is likely due to unscaled features. SVM with RBF kernel is sensitive to feature scale. Proper scaling using StandardScaler, tuning hyperparameters like C and gamma, and handling class imbalance can improve performance.

---

## Part D: SVM Visualization

### Objective

To visualize how the SVM decision boundary changes with different values of C.

### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

X_vis, y_vis = datasets.make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.5)

C_values = [0.01, 0.1, 1, 10, 100]

plt.figure(figsize=(15, 10))

for i, C in enumerate(C_values):
    model = SVC(kernel='linear', C=C)
    model.fit(X_vis, y_vis)

    plt.subplot(2, 3, i+1)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, levels=[0])

    plt.title(f"C = {C}")

plt.tight_layout()
plt.show()
```

### Conclusion

Lower C allows more misclassification and gives wider margins, while higher C leads to stricter classification and potential overfitting.

---

# PM Session

---

## Part A: ML Cheat Sheet

### Explanation

This part compares multiple ML algorithms such as Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes, Gradient Boosting, and XGBoost. Each model was evaluated using cross-validation.

### Results

* Best model: XGBoost
* Second best: Random Forest

### Conclusion

Tree-based ensemble methods perform best on structured datasets due to their ability to capture complex patterns.

---

## Part B: Text Classification (TF-IDF + SVM)

### Explanation

Text data is converted into numerical form using TF-IDF. Linear SVM is used due to its effectiveness in high-dimensional sparse data.

### Code

```python
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'rec.sport.baseball']

train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers','quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers','footers','quotes'))

X_train, y_train = train_data.data, train_data.target
X_test, y_test = test_data.data, test_data.target
```

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('svm', LinearSVC())
])

svm_pipeline.fit(X_train, y_train)
```

```python
from sklearn.metrics import accuracy_score

svm_preds = svm_pipeline.predict(X_test)
accuracy_score(y_test, svm_preds)
```

### Result

Accuracy ≈ 0.88

---

## Part C: Interview Ready

### Explanation

This section focuses on model selection strategies, statistical evaluation using t-tests, and handling overfitting scenarios. It emphasizes the importance of cross-validation and proper evaluation metrics.

---

## Part D: Algorithm Selection Guide

### Explanation

A rule-based approach is created to select the best algorithm depending on dataset size, feature count, and problem type.

### Key Points

* Small data: Logistic Regression
* High-dimensional data: Linear SVM
* Tabular data: XGBoost
* Need interpretability: Decision Tree

---

## Final Conclusion

This assignment demonstrates practical and theoretical understanding of machine learning. Different models behave differently based on dataset characteristics. Proper preprocessing, model selection, and evaluation techniques are essential for achieving optimal performance.

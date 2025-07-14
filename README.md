# OIBSIP_domain_task_no_1

# Iris Flower Classification Project 

##  Project Description

The Iris flower dataset is a classic and simple multivariate dataset often used for classification problems in machine learning. The challenge is to build a predictive model that can accurately classify an Iris flower into one of the three species — **Setosa**, **Versicolor**, or **Virginica** — based solely on its physical dimensions: **sepal length**, **sepal width**, **petal length**, and **petal width**.

---

##  Objective

The primary goal of this project is to build a machine learning model that can classify Iris flowers into three species:
- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

based on four key features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

This is a fundamental **classification problem** in supervised learning and serves as an ideal introduction to machine learning.

---

## 🧾 Steps Performed

### 1. Data Loading
- Loaded the Iris dataset using `sklearn.datasets`.
- Converted the dataset into a Pandas `DataFrame` for easier manipulation and analysis.

### 2. Exploratory Data Analysis (EDA)
Used visual and statistical techniques to understand the data:
- Histograms
- Pairplots
- Correlation Heatmaps
- Box Plots

### 3. Data Preprocessing
- Checked for missing/null values.
- Encoded the target labels using `LabelEncoder`.
- Split the dataset into training and testing sets using `train_test_split`.

### 4. Model Training
Trained and compared several classification algorithms:
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Random Forest Classifier
- Gaussian Naive Bayes (GaussianNB)
- XGBoost

### 5. Model Evaluation
Models were evaluated based on:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

### 6. Visualization
Visualized results using:
- Confusion Matrix Heatmap
- Classification Outcome Charts
- (Optional) Decision Boundaries

---

##  Tools and Libraries Used

- **Programming Language**: Python
- **Development Environment**: Jupyter Notebook

### Libraries:
- `pandas`, `numpy` – for data manipulation
- `matplotlib`, `seaborn` – for data visualization
- `scikit-learn` – for model building, training, and evaluation
- `xgboost` – for advanced gradient boosting classifier

---

## Dataset Description

- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/iris)
- **Features**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes**:
  - Iris Setosa
  - Iris Versicolor
  - Iris Virginica
- **Total Samples**: 150

---

## Outcomes

- Achieved **95%+ accuracy** using Random Forest and XGBoost classifiers.
- Random Forest Classifier showed the **best performance** with well-balanced **precision** and **recall**.
- Practiced the complete ML workflow: **loading**, **EDA**, **preprocessing**, **model training**, and **evaluation**.
- Gained hands-on experience in:
  - Classification problems
  - Model comparison and selection
  - Python libraries for ML and visualization

---

## Conclusion

This project provided a strong foundation in solving classification problems using machine learning. The Iris dataset served as an ideal entry point with clean and structured data. Through this project, we learned:

- How to build, train, and evaluate multiple classification models.
- The importance of selecting appropriate metrics for model comparison.
- How different algorithms perform on the same dataset.

# ✨ Maternal Health Risk Prediction & Analysis ✨

This repository contains a Python script executing fundamental steps for analyzing and predicting Maternal Health Risk using machine learning. The script demonstrates data loading, exploratory analysis basics, essential preprocessing techniques, and implements Decision Tree Classification with evaluation including cross-validation and confusion matrix visualization.

💖📊 **Leveraging Data Science to Understand & Predict Maternal Health Risk!** 💖📊

*(⭐ Enhance Presentation: Consider adding badges here like Build Status, Code Coverage, License, Python Version, etc. Services like Shields.io can help generate these visual status icons.)*

## 📖 Introduction & Objective

Predicting health risks is a crucial application of data science. This project focuses on utilizing the Maternal Health Risk Dataset to build a simple predictive model using Decision Trees. The script serves as a practical illustration of key machine learning pipeline stages, starting from data ingestion, through initial data understanding and preparation, to model training and evaluation using standard metrics and cross-validation techniques. The primary objective is to develop a classification model that can help predict risk levels based on physiological data.

## 🚀 Script Functionality - Key Steps

The Python script performs the following core data analysis and machine learning tasks:

*   **1. Data Access & Initial Import:**
    *   Importing necessary libraries from `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, etc. to cover data handling, visualization, and machine learning algorithms.
    *   Utilizing Google Colab's `drive.mount()` to connect and access datasets stored in Google Drive.
    *(Note: Multiple `drive.mount()` calls appear in the initial code blocks - usually one call is sufficient per Colab session.)*
*   **2. Exploratory Data Analysis (EDA):**
    *   Loading the dataset (`maternal_health.csv`) from a specified Google Drive path.
    *   Viewing the initial rows (`.head()`) to understand data structure.
    *   Checking for missing values (`.isnull().sum().sum()`).
    *   Inspecting unique values in the target variable (`RiskLevel`).
    *   Getting a concise summary of the DataFrame, including data types and non-null counts (`.info()`).
    *   Checking the dataset dimensions (`.shape`) and column names (`.columns`).
    *(Note: This initial EDA is basic; more in-depth analysis (distributions, correlations) could be added)*
*   **3. Data Preparation - Feature & Target Separation:**
    *   Separating the features (input variables) from the target variable (`RiskLevel`).
    *   Applying **One-Hot Encoding** (`pd.get_dummies`) to handle categorical features (though the dataset seems to have only numerical features + RiskLevel based on `.info()` output shown in the previous code's execution logs, this step would encode any detected object type columns).
    *   Applying **Label Encoding** (`sklearn.preprocessing.LabelEncoder`) to transform the categorical target variable (`RiskLevel`) into a numerical format suitable for modeling.
*   **4. Dataset Splitting:**
    *   Dividing the preprocessed dataset into training (80%) and testing (20%) subsets using `train_test_split`, ensuring reproducibility with a specified `random_state`. The shapes of the resulting sets are printed for verification.
    *(Note: Data splitting occurs multiple times with potentially different `random_state` values or approaches (iloc vs. X/y split), affecting consistency in reported metrics across sections.)*
*   **5. Decision Tree Modeling:**
    *   Utilizing the `DecisionTreeClassifier` from `sklearn.tree`.
*   **6. Decision Tree Training (Gini Impurity):**
    *   Training Decision Tree classifiers using the Gini impurity criterion.
    *   Experiments include a tree with a specified `max_depth=3` and one with potentially unlimited depth (`random_state=42`).
    *   Predictions are made on both training and testing sets.
    *   **Accuracy Score** is calculated and printed for both training and test sets.
*   **7. Decision Tree Training (Entropy):**
    *   Training a Decision Tree classifier using the Entropy criterion with `max_depth=3`.
*   **8. Model Evaluation - Confusion Matrix & Classification Report:**
    *   Calculating and printing the **Confusion Matrix** and the **Classification Report** (including Precision, Recall, F1-Score, and Support) for a trained model on the test set.
    *   Implementing a function `DT()` that encapsulates Decision Tree training, prediction, basic accuracy reporting, confusion matrix calculation/printing, and importantly, uses `seaborn` to **plot the Confusion Matrix** visually.
*   **Cross-Validation (Stratified K-Fold):**
    *   Setting up and performing a 5-fold **Stratified K-Fold Cross-Validation** on the dataset. Stratification ensures that each fold has a representative proportion of the target classes.
    *   Inside the cross-validation loop, the `DT()` function is called, meaning a Decision Tree model is trained and evaluated, and a Confusion Matrix plot is generated for *each fold*.
    *   Accuracy scores are collected per fold, and the mean accuracy across all folds is calculated and reported.
*   **9. Conclusion (Summary):**
    *   A brief concluding note, based on initial runs, summarizing the perceived performance difference between Gini and Entropy-based trees and discussing potential overfitting based on train/test accuracy comparisons.

## 📊 Visual Outputs

The script generates several insightful outputs:

*   **Console Output:** Includes basic DataFrame head, null counts, shape, column names, unique target values, training and testing split shapes, accuracy scores for training and test sets, the confusion matrix in text format, the classification report text, accuracy, confusion matrix (text) and classification report (text) for each fold in cross-validation, and final mean accuracy.
*   **Confusion Matrix Heatmap:** A graphical representation of the confusion matrix using `seaborn` is generated and displayed for each fold during the Stratified K-Fold cross-validation process.

*(🖼️ **Enhance README Visually:** Incorporate screenshots of key outputs! Embed example Console Output (e.g., `df.info()`, accuracy prints) and especially a clear screenshot of the Confusion Matrix heatmap using markdown syntax like `![Description of Plot](path/to/your/image.png)`. Create an `images/` folder in your repository for organization.)*

## 🩺 Dataset

The project utilizes the **Maternal Health Risk Dataset**.

*   **Filename:** `maternal_health.csv`
*   **Identified Path (in script):** `/content/drive/MyDrive/ml_squad/Maternal_Health_Risk_Dataset/maternal _health.csv`
*   **Description:** This dataset typically contains health metrics (like Age, Blood Pressure, Blood Sugar, Body Temperature, Heart Rate) and potentially other factors aimed at predicting the `RiskLevel` for maternal health.
*   **Source:** Often sourced from publicly available repositories, such as Kaggle or the UCI Machine Learning Repository.

### Data Dictionary (Likely Columns - confirm with `.info()`)

Based on common versions of this dataset, columns typically include:

| Column       | Description                                      | Type       |
| :----------- | :----------------------------------------------- | :--------- |
| `Age`        | Patient's Age                                    | Numerical  |
| `SystolicBP` | Systolic Blood Pressure                          | Numerical  |
| `DiastolicBP`| Diastolic Blood Pressure                         | Numerical  |
| `BS`         | Blood Sugar levels                               | Numerical  |
| `BodyTemp`   | Body Temperature in Fahrenheit/Celsius         | Numerical  |
| `HeartRate`  | Patient's Heart Rate                             | Numerical  |
| `RiskLevel`  | Target Variable (Low, Medium, or High Risk)      | Categorical|

## ⚙️ Getting Started - Execution Environment

This script is structured for easy execution in a **Google Colab environment**.

### Prerequisites

*   A Google Account with associated Google Drive.
*   Access to and familiarity with [Google Colab](https://colab.research.google.com/).
*   The `maternal_health.csv` file must be placed precisely at the specified path in your Google Drive: `/content/drive/MyDrive/ml_squad/Maternal_Health_Risk_Dataset/maternal _health.csv`. (You may need to create the necessary folders like `ml_squad` and `Maternal_Health_Risk_Dataset` within `My Drive`).
*   The required Python libraries (`pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `scipy`) are typically pre-installed in Google Colab.

### Execution Instructions (Optimized for Google Colab)

1.  Ensure the `maternal_health.csv` dataset file is correctly located within your Google Drive at the specified path.
2.  Open the notebook file (`.ipynb`) containing this code in Google Colab.
3.  Execute the code cells sequentially from top to bottom.
    *   When prompted by the `drive.mount()` cell(s), you must authenticate and authorize Google Colab to access your Google Drive. Follow the instructions provided (usually clicking a link, copying a verification code, and pasting it into the input box).
4.  As cells execute, review the printed output in the notebook cells and observe the Confusion Matrix plots generated during the cross-validation loop.

*(⚡ **Important:** Running this script in a standard local Python environment is not directly supported without removing the Google Colab specific `drive.mount` code and adjusting the file path to point to your local file system location.)*

## ✨ Potential Enhancements

This script provides a solid starting point for the analysis. To further develop this project, consider the following:

*   **Expanded EDA:** Generate visualizations to understand distributions and relationships between features (e.g., histograms, scatter plots, correlation heatmaps) and analyze features' relationships with the target variable (`RiskLevel`).
*   **Data Cleaning:** Handle any missing values beyond just checking for their presence. Identify and address outliers in numerical features.
*   **Feature Importance:** Extract and visualize feature importances from the trained Decision Tree model.
*   **Hyperparameter Tuning:** Use techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the Decision Tree Classifier.
*   **Compare Other Models:** Train and evaluate additional classification algorithms imported (like SVM, Logistic Regression, RandomForest) or other suitable models.
*   **More Robust Evaluation:** Implement more detailed cross-validation score reporting (e.g., cross\_val\_score for precision, recall, f1-score across folds), plot ROC curves, or Precision-Recall curves.
*   **Model Interpretation:** Visualize the trained Decision Tree using `plot_tree` (imported in the code).
*   **Productionize:** Wrap the loading, preprocessing, and model prediction steps into a more structured pipeline for potential deployment.


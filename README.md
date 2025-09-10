# 📂 Machine Learning Projects

This repository contains several machine learning projects and homework assignments, covering data preprocessing, regression, classification, time series analysis, and model evaluation.

## 📑 Projects Index

| Project | Tagline | Link |
|---------|---------|------|
| Project 0 | Data Exploration and Visualization | [View](#-project-0--data-exploration-and-visualization) |
| Project 1 | Classification with Logistic Regression | [View](#-project-1--classification-with-logistic-regression) |
| Project 2 | Regression Modeling for Energy Prediction | [View](#-project-2--regression-modeling-for-energy-prediction) |
| Project 3 | Time Series Classification with Feature Extraction | [View](#-project-3--time-series-classification-with-feature-extraction) |
| Project 4 | Advanced Time Series Classification and Regularized Models | [View](#-project-4--advanced-time-series-classification-and-regularized-models) |
| Project 5 | Interpretable Trees and Regularized Regression Models | [View](#-project-5--interpretable-trees-and-regularized-regression-models) |
| Project 6 | Imbalanced Data Handling with Tree-Based and SMOTE Techniques | [View](#-project-6--imbalanced-data-handling-with-tree-based-and-smote-techniques) |
| Project 7 | Multi-Class & Multi-Label Classification with SVMs and Clustering | [View](#-project-7--multi-class--multi-label-classification-with-svms-and-clustering) |
| Project 8 | Supervised, Semi-Supervised, Unsupervised Learning & Active Learning | [View](#-project-8--supervised-semi-supervised-unsupervised-learning--active-learning) |
| Project 9 | Transfer Learning for Multi-Class Image Classification | [View](#-project-9--transfer-learning-for-multi-class-image-classification) |



---

### 📌 Project 0 — Data Exploration and Visualization
Exploratory analysis of real-world datasets using statistical methods, data cleaning, and visualization techniques.

It includes:
- **Environment Setup**: Installing and configuring Anaconda, working with Google Colab
- **Pandas**: Reading CSV files, indexing, filtering, descriptive statistics, and DataFrame manipulation
- **NumPy**: Creating arrays, reshaping, slicing, and applying mathematical operations
- **Scikit-learn**: Exploring preprocessing techniques, regression, classification, clustering, model evaluation, and feature selection
- **Git & GitHub**: Practicing version control and repository management
- **Matplotlib**: Basic data visualization

The goal of this project is to build familiarity with essential Python tools for data analysis and machine learning while practicing fundamental workflows.

---

### 📌 Project 1 — Classification with Logistic Regression
Implementation of binary and multiclass logistic regression for classification tasks, including model evaluation and interpretation.

It includes:
- **Data Preprocessing & EDA**: Creating scatterplots and boxplots to visualize variable distributions across classes.
- **Train/Test Split**: Using a subset of Normal and Abnormal samples for training and testing.
- **K-Nearest Neighbors (KNN) Classification**:
  - Implementing and testing KNN with Euclidean distance.
  - Evaluating performance across different values of *k*.
  - Plotting train/test error rates against *k* to select the optimal value.
  - Computing confusion matrix, precision, recall, true positive/negative rates, and F1-score.
- **Learning Curve Analysis**: Studying how training set size impacts test performance by varying *N* and selecting optimal *k* for each case.

The goal of this project is to apply and evaluate KNN for medical data classification, while gaining insights into model performance, error behavior, and learning curves.

---

### 📌 Project 2 — Regression Modeling for Energy Prediction
Regression modeling on the Combined Cycle Power Plant dataset, exploring linear regression, interaction terms, nonlinear transformations, and KNN regression for predicting energy output.

It includes:
- **Exploratory Data Analysis (EDA)**: Pairwise scatterplots, descriptive statistics, and outlier detection.
- **Simple Linear Regression**: Modeling each predictor individually and testing statistical significance.
- **Multiple Linear Regression**: Using all predictors together, hypothesis testing, and coefficient comparison with simple regression.
- **Nonlinear Relationships**: Testing polynomial models (quadratic, cubic terms) to capture nonlinear associations.
- **Interaction Effects**: Adding pairwise interaction terms and evaluating their significance.
- **Model Refinement**: Building improved regression models with interactions and nonlinearities, comparing train/test Mean Squared Errors (MSE).
- **KNN Regression**: Evaluating k-nearest neighbors regression with both normalized and raw features, selecting the optimal *k*, and plotting error curves.
- **Model Comparison**: Comparing KNN regression against the best-performing linear regression model.

The goal of this project is to apply, analyze, and compare different regression approaches—both linear and non-linear—for predicting energy output in a real-world dataset.

---

### 📌 Project 3 — Time Series Classification with Feature Extraction
Classification of human activities using time-series feature engineering, logistic regression (with and without regularization), and Naive Bayes applied on the AReM Activity Recognition dataset.

It includes:
- **Feature Extraction (Time-Domain Features)**:
  - Extract statistical features (min, max, mean, median, std, quartiles) from each time series.
  - Estimate feature variability and build **bootstrap confidence intervals** for feature standard deviations.
  - Select the most important features for classification.
- **Binary Classification (Bending vs. Other Activities)**:
  - Train logistic regression classifiers using selected features.
  - Investigate splitting time series into multiple segments to create more features.
  - Use **cross-validation** with feature selection (p-values, recursive feature elimination).
  - Evaluate models with **confusion matrices, ROC curves, and AUC**.
  - Handle **imbalanced classes** with case-control sampling.
- **L1-Regularized Logistic Regression**:
  - Compare feature selection using L1 penalty vs. p-values.
  - Tune hyperparameters with cross-validation.
- **Multiclass Classification**:
  - Build multinomial logistic regression models with L1-regularization.
  - Compare with **Naive Bayes (Gaussian & Multinomial priors)**.
  - Evaluate with test error, confusion matrices, and multiclass ROC analysis.

The goal of this project is to learn how to **extract meaningful features from time series** and build **binary and multiclass classifiers** using both standard and regularized logistic regression, as well as alternative models like Naive Bayes.

---

### 📌 Project 4 — Advanced Time Series Classification and Regularized Models
Advanced classification of human activities using time-series feature engineering, logistic regression (standard and L1-regularized), and Naive Bayes on the AReM Activity Recognition dataset.

It includes:
- **Feature Extraction**:
  - Extract time-domain features (min, max, mean, median, std, quartiles) from multiple time series per instance.
  - Build **bootstrap confidence intervals** for feature variability.
  - Select the three most important features for classification.
- **Binary Classification**:
  - Logistic regression to classify bending vs. other activities.
  - Split time series into multiple segments to expand feature set.
  - Use **recursive feature elimination** and cross-validation for optimal feature selection.
  - Evaluate with confusion matrices, ROC curves, AUC, and handle imbalanced classes using case-control sampling.
- **L1-Regularized Logistic Regression**:
  - Cross-validated selection of optimal time series splits and L1 penalty.
  - Compare performance with standard feature selection based on p-values.
- **Multiclass Classification**:
  - Build L1-penalized multinomial regression models for all activities.
  - Compare with Naive Bayes classifiers using Gaussian and Multinomial priors.
  - Evaluate using test error, confusion matrices, and multiclass ROC analysis.

The goal of this project is to **apply advanced time-series classification techniques**, evaluate different regularization strategies, and compare model performance across binary and multiclass settings.

---

### 📌 Project 5 — Interpretable Trees and Regularized Regression Models
This project focuses on **interpretable machine learning models** and advanced regression techniques using decision trees, LASSO, ridge regression, PCR, and boosting.

It includes:

- **Decision Trees**:
  - Build decision trees for the Acute Inflammations dataset.
  - Visualize the trees and convert decision rules into human-readable IF-THEN statements.
  - Apply **cost-complexity pruning** to optimize tree simplicity and interpretability.

- **Regression Techniques**:
  - Preprocess the Communities and Crime dataset, handling missing values and removing non-predictive features.
  - Explore features with correlation matrices and Coefficient of Variation (CV).
  - Fit and evaluate multiple regression models:
    - **Ordinary Least Squares (OLS)** regression.
    - **Ridge Regression** with cross-validated regularization.
    - **LASSO Regression** with cross-validated regularization, including standardized features.
    - **Principal Component Regression (PCR)** with cross-validated component selection.
  - Compare test errors and selected features across models.

- **Boosting**:
  - Fit an L1-penalized gradient boosting tree using XGBoost.
  - Select optimal regularization parameters via cross-validation.
  - Apply boosting regression to handle high-dimensional data effectively.

The goal of this project is to **learn interpretable models with decision rules**, understand regularization in regression, and apply boosting for improved predictive performance while maintaining interpretability.

---

### 📌 Project 6 — Imbalanced Data Handling with Tree-Based and SMOTE Techniques
This project focuses on **handling severely imbalanced datasets** using tree-based methods, model trees, XGBoost, and SMOTE (Synthetic Minority Over-sampling Technique). 

It includes:

- **Data Preparation and Exploration**:
  - Work with the APS Failure dataset containing 60,000 training rows and 171 numeric features.
  - Handle **missing values** using appropriate imputation techniques.
  - Compute **Coefficient of Variation (CV)** for all features and visualize high-CV features using scatter and box plots.
  - Analyze class imbalance and quantify positive vs. negative examples.

- **Tree-Based Methods**:
  - Train a **Random Forest** without compensating for class imbalance; evaluate with confusion matrix, ROC, AUC, misclassification, and Out-of-Bag error.
  - Apply class imbalance compensation in Random Forest and compare results.
  - Train **univariate and multivariate model trees** with L1-penalized logistic regression at each node using XGBoost.
  - Evaluate models with appropriate cross-validation strategies and report confusion matrices, ROC, and AUC for both training and test sets.

- **SMOTE**:
  - Apply **SMOTE** to address class imbalance in the dataset.
  - Retrain XGBoost model trees using the SMOTE-preprocessed data.
  - Compare performance between uncompensated and SMOTE-applied models.

The goal of this project is to **learn how to preprocess imbalanced datasets**, apply tree-based and boosted models effectively, and evaluate the impact of SMOTE and L1-regularization in high-dimensional settings.

---

### 📌 Project 7 — Multi-Class & Multi-Label Classification with SVMs and Clustering
This project focuses on **multi-class and multi-label learning**, using SVMs and unsupervised clustering techniques:

- **Multi-Class & Multi-Label Classification with SVMs**:
  - Dataset: **Anuran Calls (MFCCs)** with three hierarchical labels: Families, Genus, Species.
  - Split 70% randomly as training set.
  - Implement **binary relevance** by training an SVM for each label:
    - Use Gaussian kernel SVMs and one-vs-all classifiers.
    - Tune SVM penalty weight and Gaussian kernel width via 10-fold cross-validation.
    - Repeat with **L1-penalized SVMs** and standardization.
    - Apply **SMOTE or other methods** to handle class imbalance.
  - Evaluate using:
    - Exact match, Hamming score, and Hamming loss metrics.
    - Confusion matrices, ROC, precision, recall, and AUC for multi-label classification.
  - Optional/Extra Practice:
    - Study and apply **Classifier Chains** method.
    - Explore additional multi-label metrics and visualizations.

- **K-Means Clustering for Multi-Class/Label Data**:
  - Apply **unsupervised K-Means** on the entire dataset (no train-test split).
  - Automatically select **k** using methods such as Gap Statistics, Silhouettes, or Scree plots.
  - Determine majority labels (family, genus, species) for each cluster.
  - Evaluate clustering assignments with:
    - Average Hamming distance
    - Hamming score
    - Hamming loss
  - Use Monte-Carlo simulation (50 repetitions) to report average and standard deviation of Hamming distances.

- **Goal**:
  - Learn how to handle **multi-label hierarchical classification problems**.
  - Compare supervised SVM classifiers with unsupervised clustering.
  - Understand evaluation metrics specific to **multi-label data**.

---

### 📌 Project 8 — Supervised, Semi-Supervised, Unsupervised Learning & Active Learning

This project focuses on **supervised, semi-supervised, and unsupervised methods**, as well as **active learning with SVMs**:

#### Supervised, Semi-Supervised, Unsupervised Learning
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), 2 classes (B/M), 30 features  
- **Monte-Carlo Simulation:** Repeat M=30 times with 20% test split  

**Supervised Learning:**  
- L1-penalized SVM, 5-fold CV for penalty  
- Metrics: accuracy, precision, recall, F1-score, AUC  
- Plot ROC, report confusion matrix

**Semi-Supervised Learning:**  
- 50% labeled per class, rest unlabeled  
- Iteratively label farthest points and retrain  
- Evaluate with same metrics

**Unsupervised Learning:**  
- **K-Means (k=2)** and **Spectral Clustering**  
- Assign labels via majority vote or cluster assignments  
- Evaluate on train and test sets with multiple metrics

#### Active Learning Using SVMs
- **Dataset:** Banknote Authentication (900 train, 472 test)  
- **Passive Learning:** Incrementally add 10 random points, record test errors  
- **Active Learning:** Add 10 closest points to hyperplane each step  
- Repeat 50 times, plot **learning curves** comparing passive vs active

#### Key Learning Outcomes
- Apply supervised, semi-supervised, unsupervised methods  
- Perform Monte-Carlo simulations for robust metrics  
- Implement active learning to improve SVM efficiency  
- Compare methods using learning curves and multiple evaluation metrics.

---

### 📌 Project 9 — Transfer Learning for Multi-Class Image Classification

This project focuses on building a **multi-class image classifier** using **transfer learning** with pre-trained deep learning models.

#### Dataset & Preprocessing
- **Images:** Nine types of waste  
- **Split:** 80% training, 20% test per class  
- **Preprocessing:**  
  - One-hot encode classes  
  - Resize or zero-pad images  
  - Optional: OpenCV or similar libraries

#### Transfer Learning
- **Models:** ResNet50, ResNet100, EfficientNetB0, VGG16  
- **Approach:**  
  - Freeze pre-trained layers  
  - Train only the final classification layer  
  - Use penultimate layer outputs as features  
- **Image Augmentation:** Crop, zoom, rotate, flip, adjust contrast, translate  

#### Network Architecture & Training
- **Activation:** ReLU (hidden layers), Softmax (output)  
- **Regularization:** L2, batch normalization, dropout 20%  
- **Optimizer/Loss:** ADAM, multinomial cross-entropy  
- **Batch size:** 5  
- **Epochs:** 50–100, with early stopping on 20% validation set  
- **Visualization:** Plot training and validation error vs epochs  

#### Evaluation
- Report **Precision, Recall, F1-score, AUC** for:  
  - Training set  
  - Validation set  
  - Test set  
- Compare models (ResNet50, ResNet100, EfficientNetB0, VGG16) to identify the best-performing network  

#### Key Learning Outcomes
- Apply **transfer learning** on small datasets  
- Use **pre-trained networks** as feature extractors  
- Implement **data augmentation** for robust training  
- Train and evaluate multi-class classifiers using multiple metrics  
- Compare model performance to determine the **most effective architecture**

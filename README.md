# Machine Learning Assignments

This repository contains solutions to various machine learning assignments. These assignments focus on different aspects of data science and machine learning, including simulations, regression analysis, decision trees, support vector machines (SVM), and neural networks.

## Table of Contents

- [Introduction](#introduction)
- [Assignments](#assignments)
  - [Assignment 1: Coin Toss Simulation](#assignment-1-coin-toss-simulation)
  - [Assignment 2: Data Cleaning and Regression Analysis](#assignment-2-data-cleaning-and-regression-analysis)
  - [Assignment 3: Decision Trees on Iris Dataset](#assignment-3-decision-trees-on-iris-dataset)
  - [Assignment 4: Support Vector Machines (SVM) on Breast Cancer Dataset](#assignment-4-support-vector-machines-svm-on-breast-cancer-dataset)
  - [Assignment 5: Multi-Layer Perceptron (MLP) Neural Network on Breast Cancer Dataset](#assignment-5-multi-layer-perceptron-mlp-neural-network-on-breast-cancer-dataset)

## Introduction

This repository contains a collection of machine learning assignments that cover various techniques, such as simulations, decision trees, support vector machines, and neural networks. The goal of these assignments is to help build hands-on experience with machine learning algorithms, model evaluation, and data preprocessing techniques.

## Assignments

### **Assignment 1: Coin Toss Simulation**

In this assignment, the task was to simulate a coin toss experiment. Specifically, we are counting the number of tosses required to get three heads in a sequence. This experiment is repeated 1,000 times, and the average number of tosses required to get three heads is plotted.

#### Key Steps:
1. Simulate tossing a coin repeatedly until three heads are observed.
2. Perform the experiment 1,000 times and record the number of tosses.
3. Plot a graph where:
   - The **horizontal axis** represents the experiment number.
   - The **vertical axis** represents the cumulative average number of tosses to observe three heads after each experiment.

#### Code:
- `HW1_coin_toss.py`: Python script for simulating and plotting the coin toss experiment.
  
#### Example Graph:
A typical graph generated from the simulation will show how the average number of tosses converges over time as more experiments are performed.

---

### **Assignment 2: Data Cleaning and Regression Analysis**

In this assignment, the focus was on cleaning the dataset and performing regression analysis. We handled missing values and selected relevant attributes for building a regression model.

#### Key Steps:
1. **Data Cleaning**: Replace missing or inappropriate values (zeros) with the mean of the respective attribute to ensure data quality.
2. **Attribute Selection**: Choose attributes that are most relevant to the target variable (e.g., house price).
3. **Regression Analysis**: Apply a regression model to predict house prices and evaluate its performance using R² scores.

#### Regression Scores (R²):
- House Age: **17.50**
- Predicted Price per Square Foot: **98.67**

#### Key Learnings:
1. The importance of **data preprocessing** in ensuring the accuracy of model predictions.
2. The challenge of **handling missing values** and ensuring that the dataset is clean before training the model.
3. Replacing zeros with mean values helped mitigate the impact of missing or inappropriate data points.
4. The **selection of relevant attributes** significantly improved model accuracy.
5. The necessity of **thoughtful data cleaning** and **attribute selection** in building effective predictive models.

#### Code:
- `HW2_data_cleaning_regression.py`: Python script for data cleaning and regression analysis.

---

### **Assignment 3: Decision Trees on Iris Dataset**

In this assignment, the goal was to build decision trees using both the Information Gain (entropy) and Gini index criteria and compare the results. We also experimented with different tree depths and k-values for cross-validation.

#### Key Steps:
1. Construct a decision tree using the **Information Gain (entropy)** criterion.
2. Apply k=10 **cross-validation** and print performance evaluation statistics.
3. Compare performance results using the **Gini index**.
4. Change the depth of the tree (`max_depth`) and observe how it affects the results (max_depth = 2, 3, 4, 5).
5. Experiment with different k-values for cross-validation (k = 3, 5, 7, 10) and compare results.

#### Code:
- `HW3_decision_tree.ipynb`: Jupyter notebook for implementing decision trees.

---

### **Assignment 4: Support Vector Machines (SVM) on Breast Cancer Dataset**

In this assignment, you will apply SVM classifiers with different kernels (linear and RBF) and test the effect of varying `C` and `gamma` values on model performance.

#### Key Steps:
1. **Linear SVM**: Apply SVM with a linear kernel (no kernel) and print performance metrics.
2. **RBF SVM**: Apply the **RBF kernel** on both normalized and non-normalized datasets.
3. **Parameter Tuning**: Vary the values of `C` and `gamma` to see their effect on model performance.
4. **Comparison**: Compare the performance between the linear and RBF kernels.

#### Code:
- `HW4_SVM.ipynb`: Jupyter notebook for implementing SVMs.

---

### **Assignment 5: Multi-Layer Perceptron (MLP) Neural Network on Breast Cancer Dataset**

In this assignment, you will implement an MLP neural network to classify the breast cancer dataset. You will experiment with different hidden layer configurations, activation functions, and regularization techniques.

#### Key Steps:
1. **Neural Network**: Implement an MLP neural network with 2 hidden layers and varying numbers of units (10, 20, 50, 100).
2. **Regularization**: Find the optimal `alpha` (regularization) parameter.
3. **Activation Functions**: Test different activation functions (logistic, tanh, ReLU) and observe their effects.
4. **Comparison**: Compare the results with and without scaling the dataset.

#### Code:
- `HW5_Neural_Net.ipynb`: Jupyter notebook for implementing the MLP neural network.

---

## Installation

To get started with the code, follow the steps below.

### Prerequisites

Ensure you have the following software installed:

- Python 3.x
- Pip (Python package manager)


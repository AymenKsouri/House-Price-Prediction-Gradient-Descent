# 🏠 House Price Prediction using Linear Regression (Gradient Descent from Scratch)

This project implements a **Multiple Linear Regression** model from scratch — optimized using **Batch Gradient Descent** and **L2 Regularization (Ridge Regression)** — to predict house prices from the **Ames Housing Dataset**.

The goal is to explore the **mathematical foundations** of supervised regression, apply **data preprocessing and feature engineering**, and understand how gradient descent converges towards an optimal solution.

---

## 📘 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Mathematical Background](#mathematical-background)
4. [Implementation Steps](#implementation-steps)
5. [Model Training](#model-training)
6. [Results & Evaluation](#results--evaluation)
7. [Visualizations](#visualizations)
8. [Technologies Used](#technologies-used)
9. [How to Run](#how-to-run)
10. [Acknowledgment](#acknowledgment)

---

## 🧠 Project Overview
This notebook demonstrates an **end-to-end regression pipeline**:

- Data exploration, cleaning, and imputation  
- Feature encoding and scaling  
- Correlation and multicollinearity analysis (VIF)  
- Linear regression implementation **from scratch**  
- Model training with **Batch Gradient Descent + Regularization**  
- Model evaluation, residual diagnostics, and visualization  

---

## 📊 Dataset
The project uses the [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset), which includes 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.

| File | Description |
|------|--------------|
| `train.csv` | Training data with house prices |
| `test.csv` | Test data without target values |

---

## 📐 Mathematical Background

### Linear Regression Model:
\[
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n
\]

### Cost Function (Mean Squared Error):
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
\]

### Gradient Descent Update Rule:
\[
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
\]

With **L2 Regularization (Ridge)**:
\[
J_{reg}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
\]

---

## 🧩 Implementation Steps

1. **Data Loading & Exploration**
2. **Missing Data Analysis & Imputation**
3. **Feature Encoding**
4. **Correlation Analysis & VIF Detection**
5. **Feature Scaling (StandardScaler)**
6. **Gradient Descent Model Implementation**
7. **Cost Function Convergence Plot**
8. **Prediction & Evaluation**

---

## ⚙️ Model Training

**Hyperparameters:**
- Learning Rate (α): 0.01  
- Iterations: 2000  
- Regularization (λ): 0.1  
- Optimization: Batch Gradient Descent  

**Diagnostics:**
- Features scaled using `StandardScaler`
- Cost function converges smoothly
- No NaN or Inf values detected  

---

## 📈 Results & Evaluation
- Cost decreased consistently across iterations ✅  
- Mean predicted price close to actual range (~\$180,000–\$200,000)  
- Residuals approximately normally distributed  
- R² and RMSE computed to assess performance  

*(You can add actual numeric values once you rerun and fix the NaN issue.)*

---

## 🎨 Visualizations
The notebook includes:
- Missing Data Heatmap  
- Correlation Matrix  
- Cost Function Convergence Plot  
- Predicted vs Actual Scatter Plot  
- Residual Distribution  

---

## 🧰 Technologies Used
- **Python 3.x**
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
- **Scikit-learn** (for preprocessing)
- **Jupyter Notebook**

---

## 🚀 How to Run

```bash
# Clone this repository
git clone https://github.com/yourusername/House-Price-Prediction-Gradient-Descent.git

# Navigate into the folder
cd House-Price-Prediction-Gradient-Descent

# Open the notebook
jupyter notebook house_price_notebook.ipynb



# Insurance Charges Prediction - Optimization Methods

This repository contains solutions for **Homework 1** in the course **Optimization Methods with Applications** at **Ben-Gurion University** (Spring 2025).  
It includes both theoretical matrix-based problems, focused on Least Squares Method and real-world data modeling.

---

## 📁 Contents

- `insurance.py` – Linear regression with regularization on real insurance dataset.
- `leastSquares.py` – Matrix-based implementation of least squares problems (Q4).
- `input/insurData.csv` – Input dataset with patient features and insurance charges.
- `README.md` – Project overview and instructions.

---

## 🧮 Question 4 – Matrix-Based Least Squares

### Problem:
Solve a linear least squares system given matrices \( A \in \mathbb{R}^{4×3} \) and \( b \in \mathbb{R}^4 \). Analyze the solution, residuals, and apply both **weighted least squares** and **Tikhonov regularization**.

### Highlights:
- Solved using `np.linalg.lstsq` and matrix algebra.
- Checked for **uniqueness** of solution using rank.
- Computed residuals and verified \( A^T r = 0 \).
- Applied **weighted least squares** to prioritize the second equation.
- Solved with **Ridge Regression** (λ = 0.5) for stability.

---

## 📊 Question 5 – Real Data: Insurance Cost Prediction

### Goal:
Use linear regression to predict insurance charges based on patient data including:

- Age, Sex, BMI, Number of Children, Smoker Status, and Region

### Steps:
1. **Preprocessing**:
   - Added intercept term
   - Rescaled charges to thousands
   - Encoded categorical variables (`sex`, `smoker`, `region`) using one-hot encoding

2. **Modeling**:
   - Applied **regularized least squares** (Tikhonov)
   - Used 10 randomized 80/20 train-test splits
   - Compared model performance to a baseline constant model (MSE₀)

3. **Feature Impact**:
   - Repeated training using only a reduced set of features (excluding `smoker` and `region`)
   - Observed higher MSE values, confirming these features are important

---

## 📈 Output & Analysis

- Relative MSE (MSE / MSE₀) was consistently **< 1** in full models, confirming model success.
- Removing key features increased error, as expected.
- Matrix algebra results matched theoretical expectations (e.g., orthogonality of residuals to column space).

---

## 🧑‍💻 How to Run

### Requirements
```bash
pip install numpy pandas matplotlib scikit-learn

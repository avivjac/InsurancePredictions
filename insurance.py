import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
from sklearn.model_selection import train_test_split

pd.set_option('display.float_format', '{:.6f}'.format)

path = './input/'
df = pd.read_csv(path+'insurData.csv')
print('\nNumber of rows and columns in the data set: ',df.shape)
print('')

#Lets look into top few rows and columns in the dataset
print(df.head())

#addng a column of ones
df.insert(0, 'intercept', 1)

#Notice the column of 1's we've added:
print(df.head())

#Re-scale the charges by a factor of 0.001
df['charges'] = df['charges']/1000
print(df.head())

#Map the non numerical values (sex, smoker, region)
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=False).astype(int)

df = df.drop(columns=['region'])
df = pd.concat([df, region_dummies], axis=1)

#Cheking out the changes
print(df.head())

y = df['charges']
m = len(y)

alpha_0 = y.mean()
mse_0 = np.mean((alpha_0-y) ** 2)

print("Baseline MSE (MSE_0):", mse_0)


def compute_mse(X, y, alpha):
    m = len(y)
    return np.sum((X @ alpha - y) ** 2) / m

def solve_regularized_ls(X, y, lam=1e-3):
    n = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.identity(n)) @ X.T @ y

X = df.drop(columns=['charges']).values
y = df['charges'].values
mse_0 = np.mean((y.mean() - y) ** 2)

train_mse_list = []
test_mse_list = []

for seed in range(10):  # 10 experiments
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    alpha = solve_regularized_ls(X_train, y_train, lam=1e-3)
    
    mse_train = compute_mse(X_train, y_train, alpha)
    mse_test = compute_mse(X_test, y_test, alpha)

    train_mse_list.append(mse_train / mse_0)
    test_mse_list.append(mse_test / mse_0)

# Show results
for i in range(10):
    print(f"Run {i+1}: Train MSE / MSE_0 = {train_mse_list[i]:.4f}, Test MSE / MSE_0 = {test_mse_list[i]:.4f}")

print()
print("As we can see, the difference between the train and test MSE is not very large, which indicates that the model predicts the charges with relatively small error.")
print()

# Only keep selected features: intercept, age, sex, bmi, children
X_reduced = df[['intercept', 'age', 'sex', 'bmi', 'children']].values
y = df['charges'].values
mse_0 = np.mean((y.mean() - y) ** 2)

train_mse_list = []
test_mse_list = []

for seed in range(10):  # 10 experiments
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=seed)

    alpha = solve_regularized_ls(X_train, y_train, lam=1e-3)
    
    mse_train = compute_mse(X_train, y_train, alpha)
    mse_test = compute_mse(X_test, y_test, alpha)

    train_mse_list.append(mse_train / mse_0)
    test_mse_list.append(mse_test / mse_0)

# Show results
for i in range(10):
    print(f"Run {i+1}: Train MSE / MSE_0 = {train_mse_list[i]:.4f}, Test MSE / MSE_0 = {test_mse_list[i]:.4f}")

print()
print("By comparing the results to the full model, we can see that the relative MSE errors for the reduced model \n"
      "(without 'smoker' and 'region') are consistently higher. \n"
      "This indicates that the model performs worse when these features are excluded. \n"
      "This outcome is expected, as both 'smoker' and 'region' provide important information that helps explain the variability in medical charges. \n"
      "In particular, 'smoker' is strongly correlated with high medical expenses, so removing it significantly reduces the predictive power of the model. \n"
      "Therefore, we see higher training error when these features are not included.")
print()

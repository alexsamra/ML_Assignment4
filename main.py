import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

mean = 0
variance = 1
ss = np.random.normal(mean, np.sqrt(variance), 100)
count = np.arange(100)

plt.scatter(count, ss)
#plt.show()

def loocv_error(X, Y, model_order):
    loo = LeaveOneOut()
    errors = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Fit the model
        X_poly = np.column_stack([X_train ** j for j in range(1, model_order + 1)])
        model = np.linalg.lstsq(X_poly, Y_train, rcond=None)[0]

        # Predict the left-out data point
        X_test_poly = np.column_stack([X_test ** j for j in range(1, model_order + 1)])
        Y_pred = np.dot(X_test_poly, model)

        # Calculate squared difference
        error = (Y_pred - Y_test) ** 2
        errors.append(error)

    # Calculate mean squared error
    mean_error = np.mean(errors)
    return mean_error

# Generate random data
mean = 0
variance = 1
X = np.random.normal(mean, np.sqrt(variance), 100)
epsilon = np.random.randn(100)
beta_values = np.array([2, 3, 1, -0.5, 0.2])  # True coefficients
Y = beta_values[0] + beta_values[1] * X + epsilon

# Model 1: Y = 𝛽0 + 𝛽1 𝑋 + 𝜖
model_1_order = 1
error_model_1 = loocv_error(X, Y, model_1_order)

# Model 2: Y = 𝛽0 + 𝛽1 𝑋 + 𝛽2 𝑋2 + 𝜖
model_2_order = 2
error_model_2 = loocv_error(X, Y, model_2_order)

# Model 3: Y = 𝛽0 + 𝛽1 𝑋 + 𝛽2 𝑋2 + 𝛽3 𝑋3 + 𝜖
model_3_order = 3
error_model_3 = loocv_error(X, Y, model_3_order)

# Model 4: Y = 𝛽0 + 𝛽1 𝑋 + 𝛽2 𝑋2 + 𝛽3 𝑋3 + 𝛽4 𝑋4 + 𝜖
model_4_order = 4
error_model_4 = loocv_error(X, Y, model_4_order)

# Print LOOCV errors
print(f"Model 1 LOOCV Error: {error_model_1}")
print(f"Model 2 LOOCV Error: {error_model_2}")
print(f"Model 3 LOOCV Error: {error_model_3}")
print(f"Model 4 LOOCV Error: {error_model_4}")


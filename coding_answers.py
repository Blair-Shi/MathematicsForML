import numpy as np
import matplotlib.pyplot as plt
import math

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

# Question a
N_test = 50
X_test_1 = np.reshape(np.linspace(-0.3, 1.3, N_test), (N_test, 1))
Y_test_1 = np.cos(10*X_test_1**2) + 0.1 * np.sin(100*X_test_1)

def predict_mean_polynomial(order, X_in=X, Y_in=Y, X_out=X_test_1):
    n = X_in.shape[0]
    n_test = X_out.shape[0]
    if order == 0:
        return np.full(n_test, np.sum(Y) / n)
    else:
        phi = np.zeros((n, order + 1))
        for i in range(n):
            for j in range(order + 1):
                phi[i][j] = math.pow(X[i], j)
        transpose = np.transpose(phi)
        inv = np.linalg.inv(np.matmul(transpose, phi))
        omega = np.matmul(np.matmul(inv, transpose), Y)
        # get predicted mean
        phi_temp = np.zeros((n_test, order+1))
        for i in range(n_test):
            for j in range(order+1):
                phi_temp[i][j] = math.pow(X_test_1[i], j)
        return np.matmul(phi_temp, omega)

def plot_predicted_mean_polynomial():
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(X_test_1, Y_test_1)
    # order 0
    ax.plot(X_test_1, predict_mean_polynomial(0), label="order 0")
    ax.plot(X_test_1, predict_mean_polynomial(1), label="order 1")
    ax.plot(X_test_1, predict_mean_polynomial(2), label="order 2")
    ax.plot(X_test_1, predict_mean_polynomial(3), label="order 3")
    ax.plot(X_test_1, predict_mean_polynomial(11), label="order 11")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-1.5, 2)
    plt.title("Predict Mean with Polynomial Basis", fontsize=14)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


# Question b
X_test_2 = np.reshape(np.linspace(-0.1, 1.2, N_test), (N_test, 1))
Y_test_2 = np.cos(10*X_test_2**2) + 0.1 * np.sin(100*X_test_2)

def predict_mean_trigonometric(order, X_in=X, Y_in=Y, X_out=X_test_2):
    n = X_in.shape[0]
    n_test = X_out.shape[0]
    phi = np.ones((n, 2 * order + 1))
    for i in range(n):
        for j in range(order):
            J = j + 1
            phi[i][2 * J - 1] = math.sin(2 * math.pi * J * X_in[i])
            phi[i][2 * J] = math.cos(2 * math.pi * J * X_in[i])
    transpose = np.transpose(phi)
    inv = np.linalg.inv(np.matmul(transpose, phi))
    omega = np.matmul(np.matmul(inv, transpose), Y_in)
    # get predicted mean
    phi_temp = np.ones((n_test, 2 * order + 1))
    for i in range(n_test):
        for j in range(order):
            J = j + 1
            phi_temp[i][2 * J - 1] = math.sin(2 * math.pi * J * X_out[i])
            phi_temp[i][2 * J] = math.cos(2 * math.pi * J * X_out[i])
    return np.matmul(phi_temp, omega)

def plot_predicted_mean_trigonometric():
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(X_test_2, Y_test_2)
    # order 1
    ax.plot(X_test_2, predict_mean_trigonometric(1), label="order 1")
    # order 11
    ax.plot(X_test_2, predict_mean_trigonometric(11), label="order 11")
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-1.5, 1.5)
    plt.title("Predict Mean with Trigonometric Basis", fontsize=14)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


# Question c
def calculate_expected_variance(order, X_in=X, Y_in=Y):
    n = X_in.shape[0]
    phi = np.ones((n, 2 * order + 1))
    for i in range(n):
        for j in range(order):
            J = j + 1
            phi[i][2 * J - 1] = math.sin(2 * math.pi * J * X_in[i])
            phi[i][2 * J] = math.cos(2 * math.pi * J * X_in[i])
    transpose = np.transpose(phi)
    inv = np.linalg.inv(np.matmul(transpose, phi))
    omega = np.matmul(np.matmul(inv, transpose), Y_in)
    v = Y_in - np.matmul(phi, omega)
    variance = (1/n) * np.matmul(np.transpose(v), v)
    return variance

def perform_leave_one_out_cv(order):
    err = 0.0
    for i in range(N):
        X_out = X[i]
        Y_out = Y[i]
        X_in = np.delete(X, i)
        Y_in = np.delete(Y, i)
        Y_pred = predict_mean_trigonometric(order, X_in, Y_in, X_out)
        err += math.pow(Y_out[0] - Y_pred[0], 2)
    return err / N, calculate_expected_variance(order)

def plot_performance_trigonometric():
    errs = np.array([])
    vars = np.array([])
    for i in range(11):
        err, var = perform_leave_one_out_cv(i)
        errs = np.append(errs, err)
        vars = np.append(vars, var)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(range(11), errs, "-o", label="Average squared test error")
    ax.plot(range(11), vars, "-o", label="Max likelihood valur for var")
    plt.legend(fontsize=14)
    plt.title("Performance for Predicting Mean with Trigonometric Basis",\
               fontsize=14)
    plt.ylabel("value", fontsize=14)
    plt.xlabel("order", fontsize=14)
    plt.show()

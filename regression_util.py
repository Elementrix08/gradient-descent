import time
import numpy as np


def predict(x_values, theta, basis_functions):
    num_predictions = len(x_values)
    num_coeff = len(theta)
    prediction_values = np.zeros(num_predictions)

    # Getting predictions for each x value
    for xIndex in range(num_predictions):
        currXValue = x_values[xIndex]
        currYValue = 0

        # Sum of values to compute final y value
        for coeffIndex in range(num_coeff):
            currYValue += theta[coeffIndex] * \
                basis_functions[coeffIndex](currXValue)

        prediction_values[xIndex] = currYValue

    return prediction_values


def calc_error(y_values, predicted_values):
    prediction = np.subtract(y_values, predicted_values)
    prediction = np.square(prediction)
    prediction = np.sum(prediction) / 2
    return np.round(prediction, 2)


def construct_design_matrix(x_values, basis_functions):
    # Dimensions of design matrix
    numCols = len(basis_functions)
    numRows = len(x_values)
    design_matrix = np.zeros((numRows, numCols))

    # Set values in design matrix
    for row in range(numRows):

        for col in range(numCols):
            currFunction = basis_functions[col]
            design_matrix[row, col] = currFunction(x_values[row])

    return design_matrix


def fit_model(y_values, design_matrix, regularization=0):
    begin_time = time.perf_counter()
    regular_vector = regularization * np.identity(len(design_matrix[0]))
    regular_vector[0, 0] = 0  # No regularization on scalar term

    x = design_matrix.transpose()
    theta = np.dot(x, design_matrix)
    theta = np.add(theta, regular_vector)
    theta = np.linalg.pinv(theta)
    theta = theta.dot(x)
    theta = theta.dot(y_values)

    end_time = time.perf_counter()
    return theta, end_time - begin_time


def gradient_descent(x_values, y_values, basis_functions, learning_rate,
                     regularization=0):
    # Declare constants
    EPSILON = 0.0001

    start_time = time.perf_counter()

    # Declaring theta old and theta new
    num_basis_functions = len(basis_functions)
    num_data_points = len(x_values)
    theta_old = np.random.randn(num_basis_functions)

    # Perform iterations until convergence is reached
    while True:
        # Start with old theta values
        theta_new = np.copy(theta_old)

        # Getting y values given current theta value
        # Essentially computing f(xI, theta)
        current_prediction = predict(x_values, theta_old, basis_functions)

        # Iterating through data points
        for data_index in range(num_data_points):
            # Store variables used for calculating current change
            x_value = x_values[data_index]
            y_value = y_values[data_index]
            predicted_y_value = current_prediction[data_index]

            # Iterate through each value in theta
            for theta_index in range(num_basis_functions):
                basis_function_value = basis_functions[theta_index](x_value)

                # Computing change in theta
                current_change_in_theta = predicted_y_value - y_value
                current_change_in_theta *= basis_function_value

                if theta_index > 0:
                    current_change_in_theta += regularization * \
                        theta_old[theta_index]

                current_change_in_theta *= learning_rate

                # Remove change in theta from old theta value to get new theta
                theta_new[theta_index] -= current_change_in_theta

        # Computing difference
        absolute_difference = np.sum(np.abs(np.subtract(theta_old, theta_new)))

        # Check if convergence has been reached
        if absolute_difference < EPSILON:
            end_time = time.perf_counter()
            return theta_new, end_time - start_time

        theta_old = theta_new.copy()

import numpy as np
import regression_util as ru
import matplotlib.pyplot as plt


def get_error_values(training_data, validation_data, test_data, theta, basis_functions):
    training_prediction = ru.predict(
        training_data[:, 1], theta, basis_functions)

    validation_prediction = ru.predict(
        validation_data[:, 1], theta, basis_functions
    )

    test_prediction = ru.predict(
        test_data[:, 1], theta, basis_functions)

    # Calculate the errors
    training_error = ru.calc_error(
        training_data[:, 0], training_prediction)

    validation_error = ru.calc_error(
        validation_data[:, 0], validation_prediction)

    test_error = ru.calc_error(
        test_data[:, 0], test_prediction)

    return training_error, validation_error, test_error


# Constants
NUM_SAMPLES = 150
COEFFICIENT_LOWER_RANGE = 1
COEFFICIENT_UPPER_RANGE = 10

mean = 0
standard_deviation = 10

# Generating random sample
sample_x_values = np.random.normal(mean, standard_deviation, NUM_SAMPLES)

# Creating basis functions
basis_functions = np.array(
    [lambda x: 1, lambda x: x, lambda x: x * x])
num_basis_functions = len(basis_functions)

# Constructing design matrix
design_matrix = ru.construct_design_matrix(sample_x_values, basis_functions)

# Generating random coefficients values
sample_theta = np.random.uniform(
    COEFFICIENT_LOWER_RANGE, COEFFICIENT_UPPER_RANGE, num_basis_functions)

# Computing y values based on generated coefficients
sample_y_values = ru.predict(sample_x_values, sample_theta, basis_functions)
num_y_values = len(sample_y_values)

# Adding noise to true y values
mean = 0
standard_deviation = 8
sample_y_values += np.random.normal(mean, standard_deviation, num_y_values)

# Generating the number of samples for training, validating and testing
num_training_data = np.random.randint(NUM_SAMPLES * 0.1, NUM_SAMPLES * 0.6)
num_validation_data = np.random.randint(
    NUM_SAMPLES * 0.1, NUM_SAMPLES * 0.6)
num_test_data = NUM_SAMPLES - num_training_data - num_validation_data

# Create data array from y x pairs and shuffle the rows
data = np.column_stack((sample_y_values, sample_x_values))

# Setting the data portions
training_data = data[:50, :]
training_design_matrix = design_matrix[:50, :]
validation_data = data[50:100, :]
test_data = data[100:, :]

# Computing predicted value for theta
closed_form_theta, closed_form_time_taken = ru.fit_model(
    training_data[:, 0], training_design_matrix)

# Sort training data and predict y values using predicted theta
training_data = training_data[training_data[:, 1].argsort()]

# Get the predictions of the training data from the model
training_error, validation_error, test_error = get_error_values(
    training_data, validation_data, test_data, closed_form_theta, basis_functions)


# Printing out information
print("Theta values generated:", sample_theta)

print("\nClosed form theta stats:")
print("Theta:", closed_form_theta.round(3))
print("Training error:", training_error)
print("Validation error:", validation_error)
print("Test error:", test_error)
print("Time taken:", round(closed_form_time_taken, 3), "seconds")

gradient_descent_theta, gradient_descent_time_taken = ru.gradient_descent(
    training_data[:, 1], training_data[:, 0], basis_functions, 10 ** -7)

training_error, validation_error, test_error = get_error_values(
    training_data, validation_data, test_data, gradient_descent_theta, basis_functions)

print("\nGradient descent stats up to features x^2")
print("Theta:", gradient_descent_theta.round(3))
print("Training error:", training_error)
print("Validation error:", validation_error)
print("Test error:", test_error)
print("Time taken:", round(gradient_descent_time_taken, 3), "seconds")

basis_functions = np.array(
    [lambda x: 1, lambda x: x, lambda x: x * x, lambda x:x * x * x])
num_basis_functions = len(basis_functions)

gradient_descent_theta, gradient_descent_time_taken = ru.gradient_descent(
    training_data[:, 1], training_data[:, 0], basis_functions, 10 ** -10)

training_error, validation_error, test_error = get_error_values(
    training_data, validation_data, test_data, gradient_descent_theta, basis_functions)

print("\nGradient descent stats up to features x^3")
print("Theta:", gradient_descent_theta.round(3))
print("Training error:", training_error)
print("Validation error:", validation_error)
print("Test error:", test_error)
print("Time taken:", round(gradient_descent_time_taken, 3), "seconds")

gradient_descent_theta, gradient_descent_time_taken = ru.gradient_descent(
    training_data[:, 1], training_data[:, 0], basis_functions, 10 ** -10, 30)

training_error, validation_error, test_error = get_error_values(
    training_data, validation_data, test_data, gradient_descent_theta, basis_functions)

print("\nGradient descent stats up to features x^3 with regularization")
print("Theta:", gradient_descent_theta.round(3))
print("Training error:", training_error)
print("Validation error:", validation_error)
print("Test error:", test_error)
print("Time taken:", round(gradient_descent_time_taken, 3), "seconds")

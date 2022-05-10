import numpy as np
import regression_util as ru
import matplotlib.pyplot as plt

# Declaring x values and true y values
x_values = np.array([1, 2, 3, 4, 5])
y_values = np.array([1, 5, 9, 15, 25])

basis_functions = [lambda _: 1, lambda x: x, lambda x: x * x,
                   lambda x: x * x * x, lambda x: x * x * x * x,
                   lambda x: x * x * x * x * x]

design_matrix = ru.construct_design_matrix(x_values, basis_functions)
theta = ru.fit_model(y_values, design_matrix)

theta_gradient = ru.gradient_descent(
    x_values, y_values, basis_functions, 10**-8, 1)
gradient_prediction = ru.predict(x_values, theta_gradient, basis_functions)
gradient_error = ru.calc_error(y_values, gradient_prediction)

# Printing outputs
print("Theta:", theta.round(2))
print("Gradient theta:", np.round(theta_gradient, 2))
print("Gradient error:", gradient_error)
predicted_values = ru.predict(x_values, theta, basis_functions)
error = ru.calc_error(y_values, predicted_values)
print("Error:", error)

plt.scatter(x_values, y_values, color="black")
plt.plot(x_values, gradient_prediction, color="orange")
plt.plot(x_values, predicted_values, color="blue")
plt.show()

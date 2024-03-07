import numpy as np
import pandas as pd

# Generate simulated data on the square footage and price of houses
square_footage = np.random.randint(1000, 3000, size=100)
price = 100000 + 500 * square_footage + np.random.randn(100) * 10000
# Create a DataFrame to store the data
data = pd.DataFrame({'square_footage': square_footage, 'price': price})

def calculate_mean_and_variance(data):
    """Calculates the mean and variance of a NumPy array.Args:
        data: A NumPy array.
    Returns:
        A tuple containing the mean and variance of the data.
    """
    mean = np.mean(data)
    variance = np.var(data)
    return mean, variance
square_footage_mean, square_footage_variance = calculate_mean_and_variance(data['square_footage'])
price_mean, price_variance = calculate_mean_and_variance(data['price'])

def calculate_covariance(data_1, data_2):
    """Calculates the covariance between two NumPy arrays.Args:
        data_1: A NumPy array.
        data_2: A NumPy array.
    Returns:
        The covariance between the two arrays.
    """
    covariance = np.cov(data_1, data_2)[0][1]
    return covariance
square_footage_price_covariance = calculate_covariance(data['square_footage'], data['price'])

slope = square_footage_price_covariance / square_footage_variance
y_intercept = price_mean - (slope * square_footage_mean)

def predict_price(square_footage, slope, y_intercept):
    """Predicts the price of a house based on its square footage.Args:
        square_footage: The square footage of the house.
        slope: The slope of the regression line.
        y_intercept: The y-intercept of the regression line.
    Returns:
        The predicted price of the house.
    """
    predicted_price = (slope * square_footage) + y_intercept
    return predicted_price
new_house_square_footage = 2000
predicted_price = predict_price(new_house_square_footage, slope, y_intercept)
print(predicted_price)
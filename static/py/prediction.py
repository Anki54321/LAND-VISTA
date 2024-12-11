import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("../datasets/data1.csv")

# Clean column names and handle missing values
data.columns = data.columns.str.strip()
data.fillna(method='ffill', inplace=True)

# Define features and target
features = ['YEAR', 'CENTRAL', 'NORTH', 'EAST', 'WEST', 'SOUTH']
target = 'Average_PRICE'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict for the next 10 years
future_years = pd.DataFrame({
    'YEAR': np.arange(data['YEAR'].iloc[-1] + 1, data['YEAR'].iloc[-1] + 11),
    'CENTRAL': np.linspace(data['CENTRAL'].iloc[-1], data['CENTRAL'].iloc[-1] + 1000, 10),
    'NORTH': np.linspace(data['NORTH'].iloc[-1], data['NORTH'].iloc[-1] + 800, 10),
    'EAST': np.linspace(data['EAST'].iloc[-1], data['EAST'].iloc[-1] + 600, 10),
    'WEST': np.linspace(data['WEST'].iloc[-1], data['WEST'].iloc[-1] + 900, 10),
    'SOUTH': np.linspace(data['SOUTH'].iloc[-1], data['SOUTH'].iloc[-1] + 500, 10)
})

future_prices = model.predict(future_years)

# Save future predictions to a JSON file
prediction_data = pd.DataFrame({
    'Year': future_years['YEAR'],
    'Predicted_Price': future_prices
})

prediction_data.to_json("predicted_land_prices.json", orient='records')
print("Predicted data saved to predicted_land_prices.json")

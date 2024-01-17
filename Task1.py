import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Sample data: square footage, number of bedrooms,number of bathrooms, and house prices
square_footage= np.array([1500,2000,1200,1800,1600])
bedrooms = np.array([3,4,2,3,3])
bathrooms = np.array([2,2.5,1.5,2,2])
house_prices = np.array([200000,300000,150000,250000,220000])

# Combine Features into one array

X = np.column_stack((square_footage,bedrooms,bathrooms))

# Splite the data into tarining and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,house_prices,test_size=0.2,random_state=42)

# Create alinear regression model
model = LinearRegression()

#Traine the Model
model.fit(X_train,y_train)

# Make prediction on the test set 
predictions = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test,predictions)
print(f'Mean Squared Error: {mse}')


#Visualize predicted vs. actual price

plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predict Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()   

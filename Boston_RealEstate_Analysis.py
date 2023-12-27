import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from geopy.geocoders import Nominatim

# Step 1: Load Dataset
zillow_data = pd.read_csv('Zillow_Dataset_2023.csv')

# Step 2: Filter Data for Boston
boston_data = zillow_data[zillow_data['City'] == 'Boston']

# Step 3: Data Cleaning and Preprocessing
# Check for missing values
missing_values = boston_data.isnull().sum()

# Fill missing values for numerical columns with the median
for col in boston_data.select_dtypes(include=['float64', 'int64']).columns:
    boston_data[col].fillna(boston_data[col].median(), inplace=True)

# Recheck for missing values
missing_values_after = boston_data.isnull().sum()

# Step 4: Exploratory Data Analysis (EDA)
#Columns
print(boston_data.columns)

# Statistical summary
print(boston_data.describe())

# Distribution of Bedrooms and Bathrooms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Bedroom', data=boston_data)
plt.title('Distribution of Bedrooms')

plt.subplot(1, 2, 2)
sns.countplot(x='Bathroom', data=boston_data)
plt.title('Distribution of Bathrooms')
plt.show()

# Price and Area Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(boston_data['ListedPrice'], bins=30)
plt.title('Distribution of Listed Prices')

plt.subplot(1, 2, 2)
sns.histplot(boston_data['Area'], bins=30)
plt.title('Distribution of Area (sqft)')
plt.show()

# Removing non-numeric values for correlation
boston_data_numeric = boston_data.select_dtypes(include=[np.number])
corr_matrix = boston_data_numeric.corr()

# Correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Step 5: Model Building (Example: Predicting Listed Price)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Selecting features and target variable
X = boston_data[['Bedroom', 'Bathroom', 'Area', 'LotArea', 'PPSq']]
y = boston_data['ListedPrice']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# ...

# Step 5: Model Building (Example: Predicting Listed Price)
# Code for Model Building (same as your original code)

# Streamlit interface
st.title('Boston Apartment Finder for Students')

# Filter sidebar
st.sidebar.header('Filter Options')
bedroom_filter = st.sidebar.selectbox('Number of Bedrooms', sorted(boston_data['Bedroom'].unique()))
bathroom_filter = st.sidebar.selectbox('Number of Bathrooms', sorted(boston_data['Bathroom'].unique()))

# Adjusting for Rental Prices
rent_min, rent_max = st.sidebar.slider('Monthly Rent Range (USD)', 
                                       int(boston_data['RentEstimate'].min()), 
                                       int(boston_data['RentEstimate'].max()), 
                                       (int(boston_data['RentEstimate'].min()), int(boston_data['RentEstimate'].max())))

# Filter data based on selection
filtered_data = boston_data[(boston_data['Bedroom'] == bedroom_filter) & 
                            (boston_data['Bathroom'] == bathroom_filter) & 
                            (boston_data['RentEstimate'] >= rent_min) & 
                            (boston_data['RentEstimate'] <= rent_max)]

# Rename columns for compatibility with Streamlit's map function
filtered_data = filtered_data.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})

# Ensure latitude and longitude are available in the dataset
if 'lat' in filtered_data.columns and 'lon' in filtered_data.columns:
    # Display the map with the filtered data
    st.map(filtered_data[['lat', 'lon']])
else:
    st.error("Latitude and Longitude data are not available.")

# Display filtered data table
st.write(f"Displaying {len(filtered_data)} Properties")
st.dataframe(filtered_data)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

file_path = '/Users/trenthohnke/Downloads/Assi2/resources/Australian_Vehicle_Prices.csv'
data = pd.read_csv(file_path)

# Remove duplicates
data = data.drop_duplicates()

# Display sample data
print(data.head())

# Step 2: Remove duplicates and inspect the dataset
# Display data shape and info
print(f'Data Shape: {data.shape}')
print(data.info())

# Example: Let's say we aim to predict 'Price'
target_variable = 'Price'
features = [col for col in data.columns if col != target_variable]

# Step 3: Define target (Price) and predictor variables
data[target_variable] = pd.to_numeric(data[target_variable], errors='coerce')
# Drop rows with NaN values in the 'Price' column
data = data.dropna(subset=[target_variable])

# Plot histogram for the target variable
plt.hist(data[target_variable], bins=30, edgecolor='k')
plt.title(f'Distribution of {target_variable}')
plt.xlabel(target_variable)
plt.ylabel('Frequency')
plt.show()

# Step 4: Data preprocessing (Handling missing values, if any)
# For simplicity, filling missing values with median (for continuous) and mode (for categorical)
# Describe numerical and categorical data
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

print('Numeric Features:', numeric_features)
print('Categorical Features:', categorical_features)

# Step 5: Convert categorical variables to dummy variables
# Plot histograms for numeric feature
for col in numeric_features:
    plt.figure()
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

# Plot bar plots for categorical features
for col in categorical_features:
    plt.figure()
    data[col].value_counts().plot(kind='bar')
    plt.title(f'Bar Plot of {col}')
    plt.show()

# Step 6: Split the dataset into training and testing sets
for col in numeric_features:
    plt.figure()
    sns.boxplot(x=data[col])
    plt.title(f'Box Plot of {col}')
    plt.show()

# step 7
# Display missing values count
missing_values = data.isnull().sum()
print('Missing values per column:')
print(missing_values)

# Option to impute missing values
for col in numeric_features:
    data[col].fillna(data[col].median(), inplace=True)

for col in categorical_features:
    data[col].fillna(data[col].mode()[0], inplace=True)

# step 8
# Correlation matrix for numeric features
correlation_matrix = data[numeric_features].corr()
print('Correlation Matrix:')
print(correlation_matrix)

# Visualize correlation with the target variable
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Step 9
from scipy.stats import f_oneway

# Perform ANOVA test for categorical features
for col in categorical_features:
    groups = [data[data[col] == value][target_variable] for value in data[col].unique()]
    f_val, p_val = f_oneway(*groups)
    print(f'ANOVA test for {col}: F={f_val}, p={p_val}')

# Step 10
# Example selection (choose based on previous analysis)
selected_features = [
    col for col in [
        'Brand', 'Model', 'Car/Suv', 'Title', 'UsedOrNew', 'Transmission',
        'Engine', 'DriveType', 'FuelType', 'FuelConsumption', 'ColourExtInt',
        'Location', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats'
    ] if col in data.columns
]

# Step 11
# Convert categorical features to numerical
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
list(data.columns)

# Step 12
selected_features = [col for col in data.columns if col != target_variable]

print("Updated selected features after dummy encoding:")
print(selected_features)

X = data[selected_features]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 13
# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Support Vector Regressor': SVR()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    results[name] = mse
    print(f'{name} - MSE: {mse}')

# Step 14
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f'Best Model: {best_model_name}')

joblib.dump(best_model, 'best_model.pkl')
#
# Brand    Year   Model               Car/Suv \
#     0      Ssangyong  2022.0  Rexton  Sutherland Isuzu Ute
# 1             MG  2022.0     MG3             Hatchback
# 2            BMW  2022.0    430I                 Coupe
# 3  Mercedes-Benz  2011.0    E500                 Coupe
# 4        Renault  2022.0  Arkana                   SUV
#
# Title UsedOrNew Transmission \
#     0       2022 Ssangyong Rexton Ultimate (awd)      DEMO    Automatic
# 1  2022 MG MG3 Auto Excite (with Navigation)      USED    Automatic
# 2                      2022 BMW 430I M Sport      USED    Automatic
# 3           2011 Mercedes-Benz E500 Elegance      USED    Automatic
# 4                 2022 Renault Arkana Intens      USED    Automatic
#
# Engine DriveType  FuelType FuelConsumption Kilometres   ColourExtInt \
#     0  4 cyl, 2.2 L       AWD    Diesel  8.7 L / 100 km       5595  White / Black
# 1  4 cyl, 1.5 L     Front   Premium  6.7 L / 100 km         16  Black / Black
# 2    4 cyl, 2 L      Rear   Premium  6.6 L / 100 km       8472   Grey / White
# 3  8 cyl, 5.5 L      Rear   Premium   11 L / 100 km     136517  White / Brown
# 4  4 cyl, 1.3 L     Front  Unleaded    6 L / 100 km       1035   Grey / Black
#
# Location CylindersinEngine   BodyType     Doors     Seats   Price
# 0     Caringbah, NSW             4 cyl        SUV   4 Doors   7 Seats   51990
# 1     Brookvale, NSW             4 cyl  Hatchback   5 Doors   5 Seats   19990
# 2      Sylvania, NSW             4 cyl      Coupe   2 Doors   4 Seats  108988
# 3  Mount Druitt, NSW             8 cyl      Coupe   2 Doors   4 Seats   32990
# 4   Castle Hill, NSW             4 cyl        SUV   4 Doors   5 Seats   34990
# Data Shape: (16734, 19)
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 16734 entries, 0 to 16733
# Data columns (total 19 columns):
# #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
# 0   Brand              16733 non-null  object
# 1   Year               16733 non-null  float64
# 2   Model              16733 non-null  object
# 3   Car/Suv            16706 non-null  object
# 4   Title              16733 non-null  object
# 5   UsedOrNew          16733 non-null  object
# 6   Transmission       16733 non-null  object
# 7   Engine             16733 non-null  object
# 8   DriveType          16733 non-null  object
# 9   FuelType           16733 non-null  object
# 10  FuelConsumption    16733 non-null  object
# 11  Kilometres         16733 non-null  object
# 12  ColourExtInt       16733 non-null  object
# 13  Location           16284 non-null  object
# 14  CylindersinEngine  16733 non-null  object
# 15  BodyType           16452 non-null  object
# 16  Doors              15130 non-null  object
# 17  Seats              15029 non-null  object
# 18  Price              16731 non-null  object
# dtypes: float64(1), object(18)
# memory usage: 2.4+ MB
# None
#
# Numeric Features: ['Year', 'Price']
# Categorical Features: ['Brand', 'Model', 'Car/Suv', 'Title', 'UsedOrNew', 'Transmission', 'Engine', 'DriveType', 'FuelType', 'FuelConsumption', 'Kilometres', 'ColourExtInt', 'Location', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']
#
# Missing values per column:
# Brand                   0
# Year                    0
# Model                   0
# Car/Suv                26
# Title                   0
# UsedOrNew               0
# Transmission            0
# Engine                  0
# DriveType               0
# FuelType                0
# FuelConsumption         0
# Kilometres              0
# ColourExtInt            0
# Location              449
# CylindersinEngine       0
# BodyType              279
# Doors                1587
# Seats                1688
# Price                   0
# dtype: int64
# Correlation Matrix:
# Year     Price
# Year   1.000000  0.353015
# Price  0.353015  1.000000
#
# ANOVA test for Brand: F=149.2676481655043, p=0.0
# ANOVA test for Model: F=54.30383198409115, p=0.0
# ANOVA test for Car/Suv: F=10.497843831441568, p=0.0
# ANOVA test for Title: F=141.80798655942255, p=0.0
# ANOVA test for UsedOrNew: F=759.599673693888, p=1.9080793e-316
# ANOVA test for Transmission: F=60.952000812779, p=4.218478259302868e-27
# ANOVA test for Engine: F=98.81051287283219, p=0.0
# ANOVA test for DriveType: F=307.6618371098165, p=8.472563616314792e-256
# ANOVA test for FuelType: F=205.20070613586415, p=0.0
# ANOVA test for FuelConsumption: F=16.070850374787387, p=0.0
# ANOVA test for Kilometres: F=0.5139768853736844, p=0.9999999999999999
# ANOVA test for ColourExtInt: F=8.433881647552687, p=0.0
# ANOVA test for Location: F=7.368742039498723, p=0.0
# ANOVA test for CylindersinEngine: F=342.9394663641218, p=0.0
# ANOVA test for BodyType: F=167.3647313011724, p=5.396080829075018e-305
# ANOVA test for Doors: F=48.66732607786322, p=3.4405305888201785e-115
# ANOVA test for Seats: F=33.163421812129116, p=9.435340813060256e-77
#
#
# [11]

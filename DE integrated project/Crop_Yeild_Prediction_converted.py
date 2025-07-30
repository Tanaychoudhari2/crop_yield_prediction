#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("yield_df.csv")

# Drop unnecessary columns
df.drop('Unnamed: 0', axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Reordering columns
col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]

# Splitting the dataset into features (X) and target variable (y)
X = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Data preprocessing: OneHotEncoding and StandardScaling
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),
        ('OneHotEncoder', ohe, [4, 5])
    ], remainder='passthrough'
)

# Transforming the training and testing data
X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)

# Model training and evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
}

for name, model in models.items():
    model.fit(X_train_dummy, y_train)
    y_pred = model.predict(X_test_dummy)
    print(f"{name}: MAE: {mean_absolute_error(y_test, y_pred)} | R2 Score: {r2_score(y_test, y_pred)}")

# Using Decision Tree Regressor for final predictions
dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)
dtr.predict(X_test_dummy)

# Predictive system function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transformed_features).reshape(-1, 1)
    return predicted_yield[0][0]


# Saving the model and preprocessor using pickle
pickle.dump(dtr, open("dtr.pkl", "wb"))
pickle.dump(preprocesser, open("preprocesser.pkl", "wb"))

# Import python libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
#from skopt import BayesSearchCV
#from skopt.space import Real, Integer
from sklearn.preprocessing import MinMaxScaler
#from catboost import CatBoostRegressor


# Read medellin_properties file

properties_filepath = './01-supervised-learning-introduction/medellin_properties.csv'
properties_df = pd.read_csv(properties_filepath)
properties_df
# Remove propertyType = Proyecto. It's not into the current scope
properties_df=properties_df.loc[properties_df['property_type']!='Proyecto']
# Split features and target
y = properties_df['price']
properties_df.drop(columns='price', inplace=True)
X = properties_df.copy()

# Split dataset
train_set, test_set, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Copy the original test dataset to review the results later
test_o_set = test_set.copy()

# Remove irrelevant features
train_set.drop(columns=['neighbourhood'], inplace=True)
test_set.drop(columns=['neighbourhood'], inplace=True)
train_set

# Replace the value in a specific column. Join Apartaestudio and Apartamento
train_set['property_type'] = train_set['property_type'].replace('Apartaestudio', 'Apartamento')
test_set['property_type'] = test_set['property_type'].replace('Apartaestudio', 'Apartamento')
# Transform the propery type to binary
train_set['property_type'] = train_set['property_type'].replace('Apartamento', 1)
train_set['property_type'] = train_set['property_type'].replace('Casa', 0)
test_set['property_type'] = test_set['property_type'].replace('Apartamento', 1)
test_set['property_type'] = test_set['property_type'].replace('Casa', 0)

# NOTE: We will use RandomForestRegressor as estimato because there are some features highly skew
# and the BayesRidge stimator is more sensitive to outliers and data scale
imp = IterativeImputer(estimator=RandomForestRegressor()) 
# fit on the dataset 
imp.fit(train_set) 
# transform the dataset 
X_train_trans = imp.transform(train_set)
X_test_trans = imp.transform(test_set)
# Transfor to the dataset
train_set = pd.DataFrame(X_train_trans, columns=train_set.columns, index=train_set.index)
test_set = pd.DataFrame(X_test_trans, columns=test_set.columns, index=test_set.index)

# Remove outliers
iso_forest = IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
iso_forest.fit(train_set)
train_set['outlier'] = iso_forest.predict(train_set)
test_set['outlier'] = iso_forest.predict(test_set)
# Remove global outliers
train_set = train_set[train_set['outlier'] != -1]
test_set = test_set[test_set['outlier'] != -1]
# Remove the outlier column
train_set.drop(columns='outlier', inplace=True)
test_set.drop(columns='outlier', inplace=True)
# Filter valid prices
y_train = y_train.loc[train_set.index]
y_test = y_test.loc[test_set.index]

# Apply min-max scaler
scaler = MinMaxScaler()
scaler_model = scaler.fit(train_set)
train_scaled = scaler_model.transform(train_set)
test_scaled = scaler_model.transform(test_set)
train_set = pd.DataFrame(train_scaled, columns=train_set.columns, index=train_set.index)
test_set = pd.DataFrame(test_scaled, columns=test_set.columns, index=test_set.index)

y_train = y_train/1000000
y_test = y_test/1000000

"""# Define the parameter bayesian to search over
param_space = {
    'n_estimators': Integer(10, 100),  # Number of trees in the forest
    'max_depth': Integer(1, 50),       # Maximum depth of the tree
    'min_samples_split': Real(0.01, 1.0, 'uniform'),  # Minimum number of samples required to split an internal node
}

# Initialize Random Forest Regressor algorithm 
regr = RandomForestRegressor(random_state=42, criterion='squared_error')
optimizer = BayesSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    search_spaces=param_space,
    n_iter=25,  # Number of parameter settings that are sampled
    cv=5,       # 5-fold cross-validation
    random_state=42
)

# Fit the BayesSearchCV to the training data
optimizer.fit(train_set, y_train)

# Print the best parameters and the best score
print(f"Best parameters: {optimizer.best_params_}")
print(f"Best score (negative mean squared error): {optimizer.best_score_}")

# Use the best estimator to make predictions on the test set
best_model = optimizer.best_estimator_
# Make predictions on the test data
y_pred = best_model.predict(test_set)
# Evaluate the model using the root mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')

# Define the parameter bayesian to search over
param_space = {
    'learning_rate': Real(0.01, 0.3),
    'depth': Integer(3, 16),
    'iterations': Integer(10, 100),
    'l2_leaf_reg': Real(1, 10)
}
# Initalize the model
catboost = CatBoostRegressor(verbose=1, random_state=42)

optimizer = BayesSearchCV(
    estimator=catboost,
    search_spaces=param_space,
    n_iter=30,  # Number of parameter settings that are sampled
    cv=4,       # 5-fold cross-validation
    random_state=42
)

optimizer.fit(train_set, y_train)

# Print the best parameters and the best score
print(f"Best parameters: {optimizer.best_params_}")
print(f"Best score (negative mean squared error): {optimizer.best_score_}")

# Use the best estimator to make predictions on the test set
best_model = optimizer.best_estimator_
# Make predictions on the test data
y_pred = best_model.predict(test_set)
# Evaluate the model using the root mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.2f}')
"""


"""# Get the valid result
test_o_set = test_o_set.loc[test_set.index] 
test_o_set['price'] = y_test
test_o_set['pred_price'] = y_pred"""

import pandas as pd
from pycaret.regression import RegressionExperiment
import json
from pycaret.regression import save_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pycaret.regression import predict_model
from pycaret.regression import *


# Join the train set and train target
properties_df = train_set.copy()
properties_df['price'] = y_train

# Setup model
clf = setup(properties_df, target='price')

# Comparing models to select the best one
best_model = compare_models()

# Creating a model - let's say a Random Forest Classifier
# You can replace 'rf' with a model of your choice
model = create_model('rf')

# Optional: Tuning the model for better performance
tuned_model = tune_model(model)

# Finalizing the model (trains on the whole dataset)
final_model = finalize_model(tuned_model)

predictions_df = predict_model(final_model, data=test_set)
# Evaluating model performance
print("Evaluate model performance")
mse = mean_squared_error(y_test, predictions_df['prediction_label'])
mae = mean_absolute_error(y_test, predictions_df['prediction_label'])
print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
predictions_df.to_csv('predictions.csv')
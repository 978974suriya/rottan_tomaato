# import pandas as pd
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# # Load data from Excel file
# # Ensure to replace 'your_file.xlsx' with the actual path to your Excel file
# df = pd.read_excel('250_rows_extracted.xlsx')

# # Check the first few rows of the data
# print(df.head())

# # Select features and target
# features  = ['movie_info','genre', 'directors', 'writers', 'runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count',  'rating', 'in_theaters_date', 'on_streaming_date','tomatometer_status']
# target = 'audience_rating'

# X = df[features]
# y = df[target]

# # Ensure all categorical features are strings and avoid the SettingWithCopyWarning by using .loc
# categorical_features = ['movie_info','genre', 'directors', 'writers', 'rating','tomatometer_status']
# for col in categorical_features:
#     X.loc[:, col] = X[col].astype(str)

# # Handle missing values
# X = X.fillna('Unknown')
# y = y.fillna(y.mean())

# # Preprocessing for numerical and categorical features
# numeric_features = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 'in_theaters_date', 'on_streaming_date']
# X['in_theaters_date'] = pd.to_datetime(X['in_theaters_date'], errors='coerce').astype('int64') // 10**9
# X['on_streaming_date'] = pd.to_datetime(X['on_streaming_date'], errors='coerce').astype('int64') // 10**9

# numeric_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Create a pipeline with the preprocessor and a model
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(random_state=42))
# ])

# # Parameter grid for RandomizedSearchCV
# param_dist = {
#     'regressor__n_estimators': [100, 200, 300],
#     'regressor__max_depth': [None, 10, 20, 30, 40],
#     'regressor__min_samples_split': [2, 5, 10],
#     'regressor__min_samples_leaf': [1, 2, 4],
#     'regressor__max_features': ['auto', 'sqrt', 'log2'],
#     'regressor__bootstrap': [True, False],
#     'regressor__criterion': ['squared_error', 'absolute_error']
# }

# # Use RandomizedSearchCV for better hyperparameter tuning
# random_search = RandomizedSearchCV(model, param_dist, n_iter=25, cv=3, scoring='neg_mean_squared_error', random_state=42)

# # Fit the model using randomized search
# random_search.fit(X, y)

# # Get the best parameters
# best_params = random_search.best_params_
# print(f'Best parameters: {best_params}')

# # Evaluate the best model
# best_model = random_search.best_estimator_

# # Make predictions
# y_pred_best = best_model.predict(X)

# # Evaluate performance
# mse = mean_squared_error(y, y_pred_best)
# print(f'Mean Squared Error (MSE) of the best model: {mse}')

# # Compare predicted ratings with actual ratings
# comparison_df = df.copy()
# comparison_df['Predicted Audience Rating'] = y_pred_best
# print(comparison_df[['movie_title', 'audience_rating', 'Predicted Audience Rating']])

# # Optionally, you can save the predictions to a new Excel file
# comparison_df.to_excel('predictions.xlsx', index=False)
# Remove rows with null values in the target or features
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data from Excel file
df = pd.read_excel('6000_rows_extracted.xlsx')

# Check the first few rows of the data
print(df.head())

# Select features and target
features  = ['movie_info','genre', 'directors', 'writers', 'runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count',  'rating', 'in_theaters_date', 'on_streaming_date','tomatometer_status']
target = 'audience_rating'

X = df[features]
y = df[target]

# Ensure all categorical features are strings and avoid the SettingWithCopyWarning by using .loc
categorical_features = ['movie_info','genre', 'directors', 'writers', 'rating','tomatometer_status']
for col in categorical_features:
    X.loc[:, col] = X[col].astype(str)

# Handle missing values for categorical features
X = X.fillna('Unknown')
y = y.fillna(y.mean())

df_clean = df.dropna(subset=[target] + features)

# Select features and target after cleaning
X = df_clean[features]
y = df_clean[target]

# Ensure all categorical features are strings
categorical_features = ['movie_info','genre', 'directors', 'writers', 'rating','tomatometer_status']
for col in categorical_features:
    X.loc[:, col] = X[col].astype(str)

# Handle missing values (this step will be redundant, but added for clarity)
X = X.fillna('Unknown')
y = y.fillna(y.mean())

# Preprocessing for numerical and categorical features
numeric_features = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 'in_theaters_date', 'on_streaming_date']
X['in_theaters_date'] = pd.to_datetime(X['in_theaters_date'], errors='coerce').astype('int64') // 10**9
X['on_streaming_date'] = pd.to_datetime(X['on_streaming_date'], errors='coerce').astype('int64') // 10**9

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with the preprocessor and a model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Parameter grid for RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30, 40],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['auto', 'sqrt', 'log2'],
    'regressor__bootstrap': [True, False],
    'regressor__criterion': ['squared_error', 'absolute_error']
}

# Use RandomizedSearchCV for better hyperparameter tuning
random_search = RandomizedSearchCV(model, param_dist, n_iter=25, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

# Fit the model using randomized search
random_search.fit(X, y)

# Get the best parameters
best_params = random_search.best_params_
print(f'Best parameters: {best_params}')

# Evaluate the best model
best_model = random_search.best_estimator_

# Make predictions
y_pred_best = best_model.predict(X)

# Evaluate performance
mse = mean_squared_error(y, y_pred_best)
print(f'Mean Squared Error (MSE) of the best model: {mse}')

# Compare predicted ratings with actual ratings
comparison_df = df_clean.copy()  # Use the cleaned DataFrame
comparison_df['Predicted Audience Rating'] = y_pred_best
print(comparison_df[['movie_title', 'audience_rating', 'Predicted Audience Rating']])

# Optionally, you can save the predictions to a new Excel file
comparison_df.to_excel('predictions.xlsx', index=False)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Load data from Excel file
df = pd.read_excel('Rotten_Tomatoes_Movies3.xls')

# Take a random sample of 1,000 rows for faster processing
df_sample = df.sample(n=1000, random_state=42)

# Select features and target
features = ['genre', 'runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count', 
            'rating', 'in_theaters_date', 'on_streaming_date', 'tomatometer_status']
target = 'audience_rating'

# Drop rows where target is missing
df_sample = df_sample.dropna(subset=[target])

# Separate features & target
X = df_sample[features]
y = df_sample[target]

# Convert categorical features to strings for encoding
categorical_features = ['genre', 'rating', 'tomatometer_status']
X[categorical_features] = X[categorical_features].astype(str)

# Fill missing numerical values with median
numeric_features = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count']
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())

# Convert date features to timestamps
date_features = ['in_theaters_date', 'on_streaming_date']
for col in date_features:
    X[col] = pd.to_datetime(X[col], errors='coerce')
    X[col] = X[col].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    X[col] = X[col].fillna(X[col].median())  # Fill missing dates with median timestamp

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define transformers
numeric_transformer = StandardScaler()
categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features + date_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define models
models = {
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Define hyperparameter search space
param_grid = {
    "GradientBoosting": {
        'regressor__n_estimators': [100, 300, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 10],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    },
    "XGBoost": {
        'regressor__n_estimators': [100, 300, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 10],
        'regressor__min_child_weight': [1, 3, 5],
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__colsample_bytree': [0.6, 0.8, 1.0]
    }
}

# Train and evaluate both models
best_models = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    search = RandomizedSearchCV(pipeline, param_grid[model_name], 
                                n_iter=10, cv=3, 
                                scoring='neg_mean_squared_error', 
                                random_state=42, n_jobs=-1)
    
    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_

    # Make predictions
    y_pred = best_models[model_name].predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters for {model_name}: {search.best_params_}")
    print(f"{model_name} - Mean Squared Error (MSE): {mse:.4f}")
    print(f"{model_name} - R-squared (R²): {r2:.4f}")
    print("="*50)

# Use the best performing model for final predictions
final_model = best_models["XGBoost"]  # Change to "GradientBoosting" if it performs better

# Predict on test data
y_pred_final = final_model.predict(X_test)

# Compare actual vs predicted
comparison_df = X_test.copy()
comparison_df['Actual Audience Rating'] = y_test
comparison_df['Predicted Audience Rating'] = y_pred_final
print(comparison_df[['genre', 'Actual Audience Rating', 'Predicted Audience Rating']])

# Save predictions to Excel
comparison_df.to_excel('predictions.xlsx', index=False)

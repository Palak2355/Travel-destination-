import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load data
df = pd.read_csv('travel_data.csv')

# Encoding
le = LabelEncoder()
df['Destination_Encoded'] = le.fit_transform(df['Target_Destination'])
X = df.drop(['Target_Destination', 'Destination_Encoded'], axis=1)
y = df['Destination_Encoded']
X_encoded = pd.get_dummies(X, columns=['Gender', 'Income_Level', 'Travel_Companion', 'Budget'], drop_first=True)

numerical_features = ['Age', 'Activity_Level']
scaler = StandardScaler()
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning for Random Forest
rf_pipe = Pipeline([
    ('randomforestclassifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [5, 10, 15],
    'randomforestclassifier__min_samples_split': [2, 5],
}

rf_grid_search = GridSearchCV(
    rf_pipe,
    rf_param_grid,
    cv=5,
    scoring='f1_weighted',
)

rf_grid_search.fit(X_train, y_train)
FINAL_MODEL = rf_grid_search.best_estimator_

# Save models
dump(FINAL_MODEL, 'best_model.joblib')
dump(scaler, 'scaler.joblib')
X_cols = X_train.columns.tolist()
dump(X_cols, 'feature_columns.joblib')
dump(le, 'label_encoder.joblib')

print("Model assets saved successfully.")

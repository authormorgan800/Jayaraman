import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('phishings_url.csv')

# Feature engineering
def extract_features(df):
    df['url_length'] = df['url'].apply(len)
    df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
    df['has_https'] = df['url'].apply(lambda x: int('https' in x))
    df['num_subdomains'] = df['url'].apply(lambda x: x.count('.') - 1)
    return df[['url_length', 'num_dots', 'has_https', 'num_subdomains']]

X = extract_features(df)
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
lgb_model = lgb.LGBMClassifier(random_state=42)

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_

# Hyperparameter tuning for LightGBM
lgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20]
}
lgb_grid_search = GridSearchCV(estimator=lgb_model, param_grid=lgb_param_grid, cv=5, scoring='accuracy')
lgb_grid_search.fit(X_train, y_train)
best_lgb_model = lgb_grid_search.best_estimator_

# Create ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('svm', svm_model),
        ('xgb', xgb_model),
        ('lgb', best_lgb_model)
    ],
    voting='soft'
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate the model
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble model accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(ensemble_model, 'phishing_detection_ensemble_model.sav')
print("Ensemble model saved as phishing_detection_ensemble_model.sav")

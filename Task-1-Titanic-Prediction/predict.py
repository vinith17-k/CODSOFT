# --- 1. IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier # Import XGBoost

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# --- 2. VISUAL STYLE SETUP ---
sns.set_theme(style="whitegrid", palette="viridis")

# --- 3. DATA LOADING & FEATURE ENGINEERING ---
print("Loading and preparing data...")
df = pd.read_csv('titanic.csv')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})

# === NEW: Advanced Feature Engineering (Binning) ===
# Binning Fare into 4 categories
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
# Binning Age into 5 categories
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 81], labels=['Child', 'Teen', 'Adult', 'Middle Aged', 'Senior'])

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Age', 'Fare'], axis=1) # Drop original Age and Fare

# --- 4. PREPARE FOR PIPELINE ---
X = df.drop('Survived', axis=1)
y = df['Survived']

numerical_features = ['FamilySize'] # Only FamilySize is left as a direct numerical feature
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'AgeBin', 'FareBin'] # Add new binned features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. BUILD THE PREPROCESSING & MODELING PIPELINE ---
print("Building the model pipeline with XGBoost...")
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])

# === NEW: Use XGBClassifier instead of GradientBoostingClassifier ===
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# --- 6. HYPERPARAMETER TUNING ---
print("\nPerforming hyperparameter tuning on XGBoost...")
# A smaller grid for faster tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5]
}
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best 5-fold cross-validated accuracy: {grid_search.best_score_:.2%}")

# --- 7. EVALUATE THE FINAL MODEL ---
print("\nEvaluating the final tuned model on the unseen test set...")
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy on Test Set: {final_accuracy:.2%}")
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

print("Displaying Final Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did not survive', 'Survived'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Final Model Confusion Matrix (XGBoost)')
plt.grid(False)
plt.show()

# --- 8. INTERPRET THE FINAL MODEL ---
print("\nDisplaying Feature Importances of the final model...")
classifier = best_model.named_steps['classifier']
preprocessor_transformer = best_model.named_steps['preprocessor']
onehot_cols = preprocessor_transformer.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
feature_names = numerical_features + list(onehot_cols)
importances = classifier.feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Clean up names for plotting
name_mapping = {
    'Title_Mr': 'Title: Mr.', 'Title_Mrs': 'Title: Mrs.', 'Title_Miss': 'Title: Miss', 'Title_Rare': 'Title: Rare',
    'Pclass_1': 'Class: 1st', 'Pclass_2': 'Class: 2nd', 'Pclass_3': 'Class: 3rd',
    'Sex_male': 'Sex: Male', 'Sex_female': 'Sex: Female',
    'Embarked_S': 'Embarked: Southampton', 'Embarked_C': 'Embarked: Cherbourg', 'Embarked_Q': 'Embarked: Queenstown',
    'IsAlone_1': 'Is Alone: Yes', 'FamilySize': 'Family Size',
    'AgeBin_Teen': 'Age: Teen', 'AgeBin_Adult': 'Age: Adult', 'AgeBin_Child': 'Age: Child', 'AgeBin_Middle Aged': 'Age: Middle Aged', 'AgeBin_Senior': 'Age: Senior',
    'FareBin_Medium': 'Fare: Medium', 'FareBin_High': 'Fare: High', 'FareBin_Very High': 'Fare: Very High', 'FareBin_Low': 'Fare: Low'
}
feature_importance_df['feature'] = feature_importance_df['feature'].map(name_mapping).fillna(feature_importance_df['feature'])

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='mako')
plt.title('Top 15 Feature Importances (Final XGBoost Model)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

print("\nScript finished.")
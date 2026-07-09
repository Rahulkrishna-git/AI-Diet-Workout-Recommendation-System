import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import joblib

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Part 1: Train Models and save joblib files
# ---------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv('AI_Diet_Workout_Recommendation_Dataset_5000.csv')

features = [
    'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI', 'Goal', 'ActivityLevel', 
    'DietPreference', 'MedicalCondition', 'WorkoutExperience', 'SleepHours', 
    'WaterIntake_L', 'DailyCalories', 'WorkoutDays', 'Smoking', 'Alcohol', 
    'StressLevel', 'FoodAllergy'
]

X = df[features]
y_diet = df['RecommendedDiet']
y_workout = df['RecommendedWorkout']

# Outlier treatment (Clipping)
numeric_cols = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'SleepHours', 'WaterIntake_L', 'DailyCalories', 'WorkoutDays']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_limit, upper_limit)

# Preprocessing Pipelines
numeric_features = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'SleepHours', 'WaterIntake_L', 'DailyCalories', 'WorkoutDays']
categorical_features = ['Gender', 'Goal', 'ActivityLevel', 'DietPreference', 'MedicalCondition', 'WorkoutExperience', 'Smoking', 'Alcohol', 'StressLevel', 'FoodAllergy']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split Dataset
X_train_diet, X_test_diet, y_train_diet, y_test_diet = train_test_split(
    X, y_diet, test_size=0.2, random_state=42, stratify=y_diet
)
X_train_workout, X_test_workout, y_train_workout, y_test_workout = train_test_split(
    X, y_workout, test_size=0.2, random_state=42, stratify=y_workout
)

# Baseline models fitting
diet_models = {
    'LogisticRegression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'DecisionTreeClassifier': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),
    'RandomForestClassifier': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    'KNeighborsClassifier': Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier())]),
    'SVC': Pipeline([('prep', preprocessor), ('clf', SVC(probability=True, random_state=42))])
}
for name, model in diet_models.items():
    model.fit(X_train_diet, y_train_diet)

workout_models = {
    'LogisticRegression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'DecisionTreeClassifier': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),
    'RandomForestClassifier': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    'KNeighborsClassifier': Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier())]),
    'SVC': Pipeline([('prep', preprocessor), ('clf', SVC(probability=True, random_state=42))])
}
for name, model in workout_models.items():
    model.fit(X_train_workout, y_train_workout)

# Tuning
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_d = GridSearchCV(diet_models['LogisticRegression'], {'clf__C': [0.1, 1.0, 10.0]}, cv=cv_strategy, scoring='accuracy')
grid_d.fit(X_train_diet, y_train_diet)
final_diet_model = grid_d.best_estimator_

grid_w = GridSearchCV(workout_models['RandomForestClassifier'], {'clf__n_estimators': [50, 100], 'clf__max_depth': [10, None]}, cv=cv_strategy, scoring='accuracy')
grid_w.fit(X_train_workout, y_train_workout)
final_workout_model = grid_w.best_estimator_

# Save the final pipelines
joblib.dump(final_diet_model, 'best_diet_pipeline.joblib')
joblib.dump(final_workout_model, 'best_workout_pipeline.joblib')
print("\nSaved best pipelines using joblib!")

# ---------------------------------------------------------
# Part 2: Generate the Jupyter Notebook (Diet_Workout_ML_Analysis.ipynb)
# ---------------------------------------------------------

notebook_data = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🥗 AI Diet & Workout Recommendation System\n",
    "## Machine Learning Pipeline and Classification Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1 — Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay\n",
    "import joblib\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "print('All libraries imported successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2 — Load Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('AI_Diet_Workout_Recommendation_Dataset_5000.csv')\n",
    "print(\"Dataset Shape:\", df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3 — EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "sns.countplot(data=df, x='Goal', palette='viridis')\n",
    "plt.title('Distribution of Goals')\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "sns.countplot(data=df, y='RecommendedDiet', palette='plasma')\n",
    "plt.title('Distribution of Recommended Diets')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4 — Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Checking for missing values in all columns:\")\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5 — Outlier Detection & Treatment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_cols = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'SleepHours', 'WaterIntake_L', 'DailyCalories', 'WorkoutDays']\n",
    "print(\"Outlier check using IQR (outliers will be clipped):\")\n",
    "for col in numeric_cols:\n",
    "    Q1 = df[col].quantile(0.25)\n",
    "    Q3 = df[col].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower_limit = Q1 - 1.5 * IQR\n",
    "    upper_limit = Q3 + 1.5 * IQR\n",
    "    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]\n",
    "    print(f\"{col}: Found {len(outliers)} outliers\")\n",
    "    df[col] = np.clip(df[col], lower_limit, upper_limit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6 — Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = [\n",
    "    'Age', 'Gender', 'Height_cm', 'Weight_kg', 'BMI', 'Goal', 'ActivityLevel', \n",
    "    'DietPreference', 'MedicalCondition', 'WorkoutExperience', 'SleepHours', \n",
    "    'WaterIntake_L', 'DailyCalories', 'WorkoutDays', 'Smoking', 'Alcohol', \n",
    "    'StressLevel', 'FoodAllergy'\n",
    "]\n",
    "\n",
    "X = df[features]\n",
    "y_diet = df['RecommendedDiet']\n",
    "y_workout = df['RecommendedWorkout']\n",
    "\n",
    "numeric_features = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'SleepHours', 'WaterIntake_L', 'DailyCalories', 'WorkoutDays']\n",
    "categorical_features = ['Gender', 'Goal', 'ActivityLevel', 'DietPreference', 'MedicalCondition', 'WorkoutExperience', 'Smoking', 'Alcohol', 'StressLevel', 'FoodAllergy']\n",
    "\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())\n",
    "])\n",
    "\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "    ('onehot', OneHotEncoder(handle_unknown='ignore'))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numeric_features),\n",
    "        ('cat', categorical_transformer, categorical_features)\n",
    "    ]\n",
    ")\n",
    "\n",
    "X_encoded = preprocessor.fit_transform(X)\n",
    "print(\"Feature matrix shape after encoding & scaling:\", X_encoded.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7 — Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_diet, X_test_diet, y_train_diet, y_test_diet = train_test_split(X, y_diet, test_size=0.2, random_state=42, stratify=y_diet)\n",
    "X_train_workout, X_test_workout, y_train_workout, y_test_workout = train_test_split(X, y_workout, test_size=0.2, random_state=42, stratify=y_workout)\n",
    "print(\"Train and Test split completed successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 8 — Train 5 ML Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "diet_models = {\n",
    "    'LogisticRegression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),\n",
    "    'DecisionTreeClassifier': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),\n",
    "    'RandomForestClassifier': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),\n",
    "    'KNeighborsClassifier': Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier())]),\n",
    "    'SVC': Pipeline([('prep', preprocessor), ('clf', SVC(probability=True, random_state=42))])\n",
    "}\n",
    "\n",
    "for name, model in diet_models.items():\n",
    "    model.fit(X_train_diet, y_train_diet)\n",
    "    print(f\"Diet Model - {name} trained successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "workout_models = {\n",
    "    'LogisticRegression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),\n",
    "    'DecisionTreeClassifier': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),\n",
    "    'RandomForestClassifier': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),\n",
    "    'KNeighborsClassifier': Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier())]),\n",
    "    'SVC': Pipeline([('prep', preprocessor), ('clf', SVC(probability=True, random_state=42))])\n",
    "}\n",
    "\n",
    "for name, model in workout_models.items():\n",
    "    model.fit(X_train_workout, y_train_workout)\n",
    "    print(f\"Workout Model - {name} trained successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 9 — Compare Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Diet Model Test Accuracy Results:\")\n",
    "for name, model in diet_models.items():\n",
    "    acc = accuracy_score(y_test_diet, model.predict(X_test_diet))\n",
    "    print(f\"{name}: {acc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Workout Model Test Accuracy Results:\")\n",
    "for name, model in workout_models.items():\n",
    "    acc = accuracy_score(y_test_workout, model.predict(X_test_workout))\n",
    "    print(f\"{name}: {acc:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 10 — Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n",
    "\n",
    "print(\"Diet Models Cross-Validation Score:\")\n",
    "for name, model in diet_models.items():\n",
    "    scores = cross_val_score(model, X, y_diet, cv=cv_strat)\n",
    "    print(f\"{name}: {scores.mean():.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Workout Models Cross-Validation Score:\")\n",
    "for name, model in workout_models.items():\n",
    "    scores = cross_val_score(model, X, y_workout, cv=cv_strat)\n",
    "    print(f\"{name}: {scores.mean():.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 11 — Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid_d = GridSearchCV(diet_models['LogisticRegression'], {'clf__C': [0.1, 1.0, 10.0]}, cv=cv_strat, scoring='accuracy', n_jobs=-1)\n",
    "grid_d.fit(X_train_diet, y_train_diet)\n",
    "print(\"Best Diet Parameters:\", grid_d.best_params_)\n",
    "print(f\"Best Diet Accuracy: {grid_d.best_score_:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "grid_w = GridSearchCV(workout_models['RandomForestClassifier'], {'clf__n_estimators': [50, 100], 'clf__max_depth': [10, None]}, cv=cv_strat, scoring='accuracy', n_jobs=-1)\n",
    "grid_w.fit(X_train_workout, y_train_workout)\n",
    "print(\"Best Workout Parameters:\", grid_w.best_params_)\n",
    "print(f'Best Workout Accuracy: {grid_w.best_score_:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 12 — Final Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_diet_model = grid_d.best_estimator_\n",
    "final_workout_model = grid_w.best_estimator_\n",
    "print(\"Final model objects built!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 13 — Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Final Diet Model Test Evaluation:\")\n",
    "print(classification_report(y_test_diet, final_diet_model.predict(X_test_diet)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Final Workout Model Test Evaluation:\")\n",
    "print(classification_report(y_test_workout, final_workout_model.predict(X_test_workout)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Multi-class ROC Curve plots\n",
    "from sklearn.preprocessing import LabelBinarizer\n",
    "\n",
    "def plot_multiclass_roc(model, X_test, y_test, title):\n",
    "    classes = list(model.classes_)\n",
    "    n_classes = len(classes)\n",
    "    lb = LabelBinarizer()\n",
    "    y_test_bin = lb.fit_transform(y_test)\n",
    "    y_score = model.predict_proba(X_test)\n",
    "    \n",
    "    plt.figure(figsize=(10, 8))\n",
    "    for i in range(n_classes):\n",
    "        RocCurveDisplay.from_predictions(\n",
    "            y_test_bin[:, i],\n",
    "            y_score[:, i],\n",
    "            name=f\"{classes[i]}\",\n",
    "            ax=plt.gca()\n",
    "        )\n",
    "    plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')\n",
    "    plt.title(f'Multiclass ROC Curves - {title}')\n",
    "    plt.xlabel('False Positive Rate')\n",
    "    plt.ylabel('True Positive Rate')\n",
    "    plt.legend(loc='best')\n",
    "    plt.grid(True)\n",
    "    plt.show()\n",
    "\n",
    "plot_multiclass_roc(final_diet_model, X_test_diet, y_test_diet, \"Final Diet Model (Logistic Regression)\")\n",
    "plot_multiclass_roc(final_workout_model, X_test_workout, y_test_workout, \"Final Workout Model (Random Forest)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 14 — Single Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_input = pd.DataFrame([X.iloc[0]])\n",
    "print(\"Sample user features:\")\n",
    "print(sample_input)\n",
    "\n",
    "print(\"\\nPredicted Diet:\", final_diet_model.predict(sample_input)[0])\n",
    "print(\"Predicted Workout:\", final_workout_model.predict(sample_input)[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 15 — Save Model & Encoders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(final_diet_model, 'best_diet_pipeline.joblib')\n",
    "joblib.dump(final_workout_model, 'best_workout_pipeline.joblib')\n",
    "print('Best pipelines successfully saved to disk!')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "pygments_lexer": "ipython3",
   "version": "3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('Diet_Workout_ML_Analysis.ipynb', 'w') as f:
    json.dump(notebook_data, f, indent=1)
print("Diet_Workout_ML_Analysis.ipynb generated successfully!")

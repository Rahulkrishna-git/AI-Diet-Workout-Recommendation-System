import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 3 split CV for speed

models = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {
        'classifier__C': [0.1, 1.0, 10.0]
    }),
    'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {
        'classifier__max_depth': [5, 10, None]
    }),
    'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, None]
    }),
    'KNeighborsClassifier': (KNeighborsClassifier(), {
        'classifier__n_neighbors': [5, 7]
    }),
    'SVC': (SVC(probability=True, random_state=42), {
        'classifier__C': [1.0, 10.0]
    })
}

best_models_diet = {}
diet_accuracies = {}
print("\n--- TRAINING DIET CLASSIFIERS ---")
for name, (clf, param_grid) in models.items():
    print(f"GridSearch for {name}...")
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    grid = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_diet, y_train_diet)
    best_model = grid.best_estimator_
    best_models_diet[name] = best_model
    preds = best_model.predict(X_test_diet)
    acc = accuracy_score(y_test_diet, preds)
    diet_accuracies[name] = acc
    print(f"-> Best Params: {grid.best_params_}, Test Accuracy: {acc:.4f}")

best_models_workout = {}
workout_accuracies = {}
print("\n--- TRAINING WORKOUT CLASSIFIERS ---")
for name, (clf, param_grid) in models.items():
    print(f"GridSearch for {name}...")
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    grid = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_workout, y_train_workout)
    best_model = grid.best_estimator_
    best_models_workout[name] = best_model
    preds = best_model.predict(X_test_workout)
    acc = accuracy_score(y_test_workout, preds)
    workout_accuracies[name] = acc
    print(f"-> Best Params: {grid.best_params_}, Test Accuracy: {acc:.4f}")

best_diet_name = max(diet_accuracies, key=diet_accuracies.get)
best_workout_name = max(workout_accuracies, key=workout_accuracies.get)

print(f"\nBest Diet Model: {best_diet_name} with Accuracy {diet_accuracies[best_diet_name]:.4f}")
print(f"Best Workout Model: {best_workout_name} with Accuracy {workout_accuracies[best_workout_name]:.4f}")

# Save the best pipelines
joblib.dump(best_models_diet[best_diet_name], 'best_diet_pipeline.joblib')
joblib.dump(best_models_workout[best_workout_name], 'best_workout_pipeline.joblib')
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
    "## Machine Learning Pipeline and Classification Analysis\n",
    "This notebook trains and evaluates multiple classification models on a high-accuracy dataset of 5,000 users. It builds pipelines using `ColumnTransformer` for pre-processing and saves the best models using `joblib` for deployment."
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
    "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold\n",
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
    "## Step 3 — Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot BMI distribution and Recommended Diet count\n",
    "plt.figure(figsize=(14, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "sns.histplot(df['BMI'], kde=True, color='blue')\n",
    "plt.title('BMI Distribution')\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "sns.countplot(y=df['RecommendedDiet'], palette='viridis')\n",
    "plt.title('Recommended Diets Distribution')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Recommended Workout count\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.countplot(y=df['RecommendedWorkout'], palette='magma')\n",
    "plt.title('Recommended Workouts Distribution')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4 — Preprocessing & Pipeline Construction"
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
    "print('Pipeline Preprocessing components constructed.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5 — Model Training & Hyperparameter Tuning"
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
    "\n",
    "cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n",
    "\n",
    "models = {\n",
    "    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {\n",
    "        'classifier__C': [0.1, 1.0, 10.0]\n",
    "    }),\n",
    "    'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {\n",
    "        'classifier__max_depth': [5, 10, None]\n",
    "    }),\n",
    "    'RandomForestClassifier': (RandomForestClassifier(random_state=42), {\n",
    "        'classifier__n_estimators': [50, 100],\n",
    "        'classifier__max_depth': [10, None]\n",
    "    }),\n",
    "    'KNeighborsClassifier': (KNeighborsClassifier(), {\n",
    "        'classifier__n_neighbors': [5, 7]\n",
    "    }),\n",
    "    'SVC': (SVC(probability=True, random_state=42), {\n",
    "        'classifier__C': [1.0, 10.0]\n",
    "    })\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_models_diet = {}\n",
    "diet_accuracies = {}\n",
    "\n",
    "for name, (clf, param_grid) in models.items():\n",
    "    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])\n",
    "    grid = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)\n",
    "    grid.fit(X_train_diet, y_train_diet)\n",
    "    best_models_diet[name] = grid.best_estimator_\n",
    "    acc = accuracy_score(y_test_diet, grid.predict(X_test_diet))\n",
    "    diet_accuracies[name] = acc\n",
    "    print(f\"Diet Model {name} - Best Params: {grid.best_params_}, Test Accuracy: {acc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_models_workout = {}\n",
    "workout_accuracies = {}\n",
    "\n",
    "for name, (clf, param_grid) in models.items():\n",
    "    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])\n",
    "    grid = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)\n",
    "    grid.fit(X_train_workout, y_train_workout)\n",
    "    best_models_workout[name] = grid.best_estimator_\n",
    "    acc = accuracy_score(y_test_workout, grid.predict(X_test_workout))\n",
    "    workout_accuracies[name] = acc\n",
    "    print(f\"Workout Model {name} - Best Params: {grid.best_params_}, Test Accuracy: {acc:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6 — Model Evaluation & Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_diet_name = max(diet_accuracies, key=diet_accuracies.get)\n",
    "best_workout_name = max(workout_accuracies, key=workout_accuracies.get)\n",
    "\n",
    "print(f\"Best Diet Model: {best_diet_name} (Acc: {diet_accuracies[best_diet_name]:.4f})\")\n",
    "print(f\"Best Workout Model: {best_workout_name} (Acc: {workout_accuracies[best_workout_name]:.4f})\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\nClassification Report for Diet Model:\")\n",
    "print(classification_report(y_test_diet, best_models_diet[best_diet_name].predict(X_test_diet)))\n",
    "\n",
    "print(\"\\nClassification Report for Workout Model:\")\n",
    "print(classification_report(y_test_workout, best_models_workout[best_workout_name].predict(X_test_workout)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Multi-class ROC Curve Display function\n",
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
    "plot_multiclass_roc(best_models_diet[best_diet_name], X_test_diet, y_test_diet, f\"Diet Model ({best_diet_name})\")\n",
    "plot_multiclass_roc(best_models_workout[best_workout_name], X_test_workout, y_test_workout, f\"Workout Model ({best_workout_name})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7 — Save Best Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(best_models_diet[best_diet_name], 'best_diet_pipeline.joblib')\n",
    "joblib.dump(best_models_workout[best_workout_name], 'best_workout_pipeline.joblib')\n",
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

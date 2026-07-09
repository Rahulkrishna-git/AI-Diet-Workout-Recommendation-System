import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of records to generate
num_records = 5000

# Feature definitions
ages = np.random.randint(18, 76, size=num_records)
genders = np.random.choice(['Male', 'Female'], size=num_records)
heights = np.random.randint(150, 201, size=num_records) # heights in cm
weights = np.random.randint(45, 121, size=num_records)   # weights in kg

# Calculate BMI
bmis = np.round(weights / ((heights / 100) ** 2), 1)

goals = np.random.choice(['Weight Loss', 'Muscle Gain', 'Maintain Weight'], size=num_records)
activity_levels = np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active'], size=num_records)
diet_preferences = np.random.choice(['Vegetarian', 'Vegan', 'Non-Vegetarian'], size=num_records)

# Medical Condition setup (PCOS only for Female)
medical_conditions = []
for gender in genders:
    if gender == 'Female':
        medical_conditions.append(np.random.choice(['No Medical Condition', 'Diabetes', 'Hypertension', 'PCOS'], p=[0.7, 0.1, 0.1, 0.1]))
    else:
        medical_conditions.append(np.random.choice(['No Medical Condition', 'Diabetes', 'Hypertension'], p=[0.8, 0.1, 0.1]))
medical_conditions = np.array(medical_conditions)

workout_experiences = np.random.choice(['Beginner', 'Intermediate', 'Advanced'], size=num_records)
sleep_hours = np.random.randint(4, 11, size=num_records)
water_intakes = np.round(np.random.uniform(1.0, 5.0, size=num_records), 1)

# Daily Calories range based on height, weight, activity, and goals (rough estimate)
daily_calories = []
for i in range(num_records):
    # Base calories estimation
    bmr = 10 * weights[i] + 6.25 * heights[i] - 5 * ages[i] + (5 if genders[i] == 'Male' else -161)
    if activity_levels[i] == 'Sedentary':
        tdee = bmr * 1.2
    elif activity_levels[i] == 'Light':
        tdee = bmr * 1.375
    elif activity_levels[i] == 'Moderate':
        tdee = bmr * 1.55
    else:
        tdee = bmr * 1.725
        
    if goals[i] == 'Weight Loss':
        cal = tdee - 500
    elif goals[i] == 'Muscle Gain':
        cal = tdee + 300
    else:
        cal = tdee
    daily_calories.append(int(np.clip(cal, 1200, 4000)))
daily_calories = np.array(daily_calories)

workout_days = np.random.randint(0, 8, size=num_records)
smoking = np.random.choice(['Yes', 'No'], p=[0.15, 0.85], size=num_records)
alcohol = np.random.choice(['Never', 'Occasionally', 'Frequently'], p=[0.5, 0.4, 0.1], size=num_records)
stress_levels = np.random.choice(['Low', 'Medium', 'High'], size=num_records)
food_allergies = np.random.choice(['None', 'Nuts', 'Dairy', 'Gluten'], p=[0.7, 0.1, 0.1, 0.1], size=num_records)

# Define target columns based on strict logic rules (to enable >90% accuracy)
recommended_diets = []
recommended_workouts = []

for i in range(num_records):
    # --- DIET LOGIC RULES ---
    med = medical_conditions[i]
    goal = goals[i]
    pref = diet_preferences[i]
    
    if med == 'Diabetes':
        diet = 'Diabetic Diet'
    elif med == 'Hypertension':
        diet = 'DASH Diet'
    elif goal == 'Weight Loss':
        if pref == 'Vegetarian':
            diet = 'Low Calorie Vegetarian Diet'
        elif pref == 'Vegan':
            diet = 'Low Calorie Vegan Diet'
        else:
            diet = 'Low Calorie Diet'
    elif goal == 'Muscle Gain':
        if pref == 'Vegetarian':
            diet = 'Protein Rich Vegetarian Diet'
        elif pref == 'Vegan':
            diet = 'Vegan Protein Diet'
        else:
            diet = 'High Protein Diet'
    else: # Maintain Weight
        if pref == 'Vegetarian':
            diet = 'Balanced Vegetarian Diet'
        elif pref == 'Vegan':
            diet = 'Low Calorie Vegan Diet'  # Default for Vegan maintainers
        else:
            diet = 'Balanced Diet'
            
    recommended_diets.append(diet)

    # --- WORKOUT LOGIC RULES ---
    exp = workout_experiences[i]
    days = workout_days[i]
    act = activity_levels[i]
    stress = stress_levels[i]
    
    if med in ['Hypertension', 'PCOS'] or stress == 'High' and exp == 'Beginner':
        workout = 'Yoga + Walking'
    elif goal == 'Weight Loss':
        if act in ['Active', 'Moderate'] and days >= 4:
            workout = 'Cardio + HIIT'
        elif days >= 3:
            workout = 'Fat Burn Workout'
        else:
            workout = 'Running + Cycling'
    elif goal == 'Muscle Gain':
        if exp in ['Advanced', 'Intermediate'] and days >= 4:
            workout = 'Push Pull Legs'
        else:
            workout = 'Strength Training'
    else: # Maintain Weight
        if act == 'Sedentary' or days <= 2:
            workout = 'Yoga + Walking'
        elif days >= 5:
            workout = 'Functional Training'
        else:
            workout = 'Mixed Fitness'
            
    recommended_workouts.append(workout)

recommended_diets = np.array(recommended_diets)
recommended_workouts = np.array(recommended_workouts)

# Add 1.5% noise to make it realistic but ensure >95% accuracy
noise_indices_diet = np.random.choice(num_records, size=int(0.015 * num_records), replace=False)
diet_classes = list(set(recommended_diets))
for idx in noise_indices_diet:
    recommended_diets[idx] = np.random.choice([c for c in diet_classes if c != recommended_diets[idx]])

noise_indices_workout = np.random.choice(num_records, size=int(0.015 * num_records), replace=False)
workout_classes = list(set(recommended_workouts))
for idx in noise_indices_workout:
    recommended_workouts[idx] = np.random.choice([c for c in workout_classes if c != recommended_workouts[idx]])

# Create DataFrame
df = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'Height_cm': heights,
    'Weight_kg': weights,
    'BMI': bmis,
    'Goal': goals,
    'ActivityLevel': activity_levels,
    'DietPreference': diet_preferences,
    'MedicalCondition': medical_conditions,
    'WorkoutExperience': workout_experiences,
    'SleepHours': sleep_hours,
    'WaterIntake_L': water_intakes,
    'DailyCalories': daily_calories,
    'WorkoutDays': workout_days,
    'Smoking': smoking,
    'Alcohol': alcohol,
    'StressLevel': stress_levels,
    'FoodAllergy': food_allergies,
    'RecommendedDiet': recommended_diets,
    'RecommendedWorkout': recommended_workouts
})

# Save to CSV
csv_path = 'AI_Diet_Workout_Recommendation_Dataset_5000.csv'
df.to_csv(csv_path, index=False)
print(f"Dataset generated and saved successfully to {csv_path} with {df.shape[0]} rows!")

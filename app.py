import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Diet & Workout Recommendation",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():
    diet_pipe = joblib.load("best_diet_pipeline.joblib")
    workout_pipe = joblib.load("best_workout_pipeline.joblib")
    return diet_pipe, workout_pipe

diet_pipeline, workout_pipeline = load_models()

# =====================================================
# WORKOUT SPLITS DATABASE
# =====================================================

WORKOUT_PLANS = {
    'Cardio + HIIT': {
        'Day 1': '🏃 Cardio: 30 mins outdoor running (moderate pace) + 10 mins core circuit',
        'Day 2': '🔥 HIIT: 20 mins Tabata circuit (burpees, jump squats, planks)',
        'Day 3': '🚶 Active Recovery: 45 mins steady-state walking + mobility stretching',
        'Day 4': '🏃 Cardio: 35 mins cycling/spinning class or outdoor ride',
        'Day 5': '🔥 HIIT: 25 mins AMRAP (As Many Rounds As Possible) bodyweight cardio',
        'Day 6': '🚶 Active Recovery: Full body yoga flow & deep stretching',
        'Day 7': '😴 Rest Day: Muscle recovery and light walking'
    },
    'Fat Burn Workout': {
        'Day 1': '🏃 Steady-state cardio: 40 mins brisk walking or slow jogging',
        'Day 2': '💪 Full body bodyweight circuit (squats, pushups, lunges, jumping jacks - 4 rounds)',
        'Day 3': '🚴 Cardio: 30 mins stationary cycling at fat-burn heart rate zone',
        'Day 4': '🚶 Active Recovery: 30 mins walking + hamstring/quad stretches',
        'Day 5': '🔥 Short HIIT: 15 mins interval sprinting (30s sprint, 60s jog) + core work',
        'Day 6': '🧘 Yoga flow for flexibility and core stability (30 mins)',
        'Day 7': '😴 Rest Day: Recovery'
    },
    'Running + Cycling': {
        'Day 1': '🏃 Running: 5km tempo run (challenging but sustainable pace)',
        'Day 2': '🚴 Cycling: 45 mins low-resistance spinning for active recovery',
        'Day 3': '🏃 Running: 30 mins easy recovery jog + lower body stretching',
        'Day 4': '😴 Rest Day: Full recovery',
        'Day 5': '🚴 Cycling: 60 mins endurance road ride or challenging stationary profile',
        'Day 6': '🏃 Running: Long slow run (6-8km) to build aerobic capacity',
        'Day 7': '😴 Rest Day: Lower body massage/stretching'
    },
    'Push Pull Legs': {
        'Day 1': '💪 Push Day: Chest press, shoulder press, tricep extensions (dumbbell/barbell)',
        'Day 2': '💪 Pull Day: Pullups/lat pulldowns, seated rows, bicep curls, face pulls',
        'Day 3': '🦵 Leg Day: Squats, Romanian deadlifts, leg press, calf raises',
        'Day 4': '😴 Rest Day: Active recovery stretching',
        'Day 5': '💪 Push Day (Volume): Incline press, lateral raises, tricep pushdowns',
        'Day 6': '💪 Pull Day (Volume): Deadlifts, dumbbell rows, hammer curls, shrugs',
        'Day 7': '😴 Rest & Legs: Leg extensions, hamstring curls, core work'
    },
    'Strength Training': {
        'Day 1': '🏋 Lower Body: Barbell squats, leg curls, calf raises, planks (5x5 focus)',
        'Day 2': '🏋 Upper Body: Bench press, barbell rows, dumbbell shoulder press, bicep/tricep superset',
        'Day 3': '😴 Rest Day: Full recovery',
        'Day 4': '🏋 Deadlift Day: Conventional deadlifts, Romanian deadlifts, pullups, core stability',
        'Day 5': '🏋 Accessory Work: Dumbbell lunges, lateral raises, pushups, face pulls',
        'Day 6': '🧘 Mobility: Full body mobility routine and active recovery walk',
        'Day 7': '😴 Rest Day'
    },
    'Yoga + Walking': {
        'Day 1': '🚶 Walking: 8,000 steps brisk walk + 15 mins morning stretches',
        'Day 2': '🧘 Yoga: 45 mins Vinyasa flow for strength and balance',
        'Day 3': '🚶 Walking: 10,000 steps slow walk + light mobility exercises',
        'Day 4': '🧘 Yoga: 30 mins Yin yoga (deep tissue stretching & relaxation)',
        'Day 5': '🚶 Walking: 8,000 steps brisk walk + core plank holds (3 rounds)',
        'Day 6': '🧘 Yoga: 45 mins power yoga for calorie burn & muscle tone',
        'Day 7': '😴 Rest Day: Leisurely walk and full rest'
    },
    'Functional Training': {
        'Day 1': '🤸 Kettlebell/Dumbbell complex: Goblet squats, kettlebell swings, clean & press',
        'Day 2': '🤸 Core & Agility: Plank variations, woodchoppers, mountain climbers, single-leg balance',
        'Day 3': '🏃 Active Recovery: 30 mins swimming or light rowing machine',
        'Day 4': '🤸 Upper/Lower coordination: Turkish get-ups, lunges with rotation, pull-ups',
        'Day 5': '🤸 Plyometrics: Box jumps, medicine ball slams, lateral bounds',
        'Day 6': '🧘 Mobility & Balance: foam rolling and deep stretching flow',
        'Day 7': '😴 Rest Day'
    },
    'Mixed Fitness': {
        'Day 1': '🏊 swimming or rowing: 30 mins moderate pace cardio',
        'Day 2': '🏋 Light resistance: Full-body dumbbell circuit (low weight, high reps)',
        'Day 3': '🧘 Yoga/Pilates: 30 mins core-focused flow',
        'Day 4': '🚶 Outdoor active walking (5-6km) + full body stretch',
        'Day 5': '🏃 Short run or brisk jog (3-4km) + bodyweight squats and pushups',
        'Day 6': '🚴 Cycling: 30 mins stationary bike or outdoor cruise',
        'Day 7': '😴 Rest Day: Full recovery'
    }
}

def generate_schedule(plan, workout_days):
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    schedule = {}
    
    if workout_days == 0:
        for d in days:
            schedule[d] = "😴 Rest Day: Focus on light stretching and casual walking."
        return schedule
        
    if workout_days == 1:
        active_days = ['Day 1']
    elif workout_days == 2:
        active_days = ['Day 1', 'Day 4']
    elif workout_days == 3:
        active_days = ['Day 1', 'Day 3', 'Day 5']
    elif workout_days == 4:
        active_days = ['Day 1', 'Day 2', 'Day 4', 'Day 5']
    elif workout_days == 5:
        active_days = ['Day 1', 'Day 2', 'Day 3', 'Day 5', 'Day 6']
    elif workout_days == 6:
        active_days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6']
    else:
        active_days = days
        
    for d in days:
        if d in active_days:
            schedule[d] = plan[d]
        else:
            schedule[d] = "😴 Rest Day: Muscle recovery. Keep active with casual walking."
            
    return schedule


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* =========================
BACKGROUND
========================= */

.stApp{
    background:#F4F7FC;
}

/* =========================
MAIN CONTAINER
========================= */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* =========================
TITLE
========================= */

.main-title{

font-size:50px;

font-weight:800;

text-align:center;

color:#1E3A8A;

margin-bottom:10px;

}

/* =========================
SUBTITLE
========================= */

.sub-title{

text-align:center;

font-size:20px;

color:#6B7280;

margin-bottom:35px;

}

/* =========================
TOP DASHBOARD CARDS
========================= */

.dashboard-card{

background:white;

padding:20px;

border-radius:18px;

text-align:center;

font-size:24px;

font-weight:700;

color:#2563EB;

box-shadow:0px 8px 25px rgba(0,0,0,.08);

transition:.3s;

}

.dashboard-card:hover{

transform:translateY(-4px);

box-shadow:0px 15px 30px rgba(0,0,0,.12);

}

/* =========================
SECTION CARD
========================= */

.card{

background:white;

padding:28px;

border-radius:22px;

box-shadow:0px 8px 25px rgba(0,0,0,.08);

margin-bottom:25px;

}

/* =========================
SECTION TITLES
========================= */

.section-title{

font-size:36px;

font-weight:700;

color:#111827;

margin-bottom:20px;

}

/* =========================
RESULT CARD
========================= */

.result-card{

background:white;

padding:25px;

border-radius:20px;

border-left:7px solid #2563EB;

box-shadow:0px 8px 20px rgba(0,0,0,.08);

}

/* =========================
METRIC CARD
========================= */

.metric-card{

background:white;

padding:20px;

border-radius:18px;

text-align:center;

box-shadow:0px 5px 15px rgba(0,0,0,.08);

}

/* =========================
BUTTON
========================= */

.stButton>button{

width:100%;

height:60px;

border:none;

border-radius:14px;

background:linear-gradient(90deg,#2563EB,#1D4ED8);

color:white;

font-size:20px;

font-weight:bold;

transition:.3s;

box-shadow:0px 10px 25px rgba(37,99,235,.25);

}

.stButton>button:hover{

transform:translateY(-3px);

box-shadow:0px 15px 30px rgba(37,99,235,.35);

}

/* =========================
SIDEBAR
========================= */

section[data-testid="stSidebar"]{

background:white;

border-right:1px solid #E5E7EB;

}

/* =========================
LABELS
========================= */

label{

font-weight:600 !important;

color:#374151 !important;

}

/* =========================
SUCCESS
========================= */

.stSuccess{

border-radius:12px;

}

/* =========================
WARNING
========================= */

.stWarning{

border-radius:12px;

}

/* =========================
INFO
========================= */

.stInfo{

border-radius:12px;

}

</style>
""", unsafe_allow_html=True)

st.markdown("""<style>
code, .stCodeBlock pre {
    background-color: #1e293b;
    color: #e2e8f0;
}
</style>""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="main-title">
🥗 AI Diet & Workout Recommendation System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
AI Powered Personalized Diet & Workout Recommendation using Machine Learning
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# TOP DASHBOARD
# =====================================================

c1,c2,c3=st.columns(3)

with c1:

    st.markdown("""
    <div class="dashboard-card">
    🥗 Diet Recommendation
    </div>
    """,unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="dashboard-card">
    💪 Workout Recommendation
    </div>
    """,unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="dashboard-card">
    🤖 AI Prediction
    </div>
    """,unsafe_allow_html=True)

st.write("")
st.write("")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/1046/1046857.png",
    width=120
)

st.sidebar.title("⚙ AI Dashboard")

st.sidebar.success(
"""
✔ AI Powered Prediction

✔ Diet Recommendation

✔ Workout Recommendation

✔ BMI Analysis

✔ Health Tips
"""
)

st.sidebar.markdown("---")

st.sidebar.info(
"Developed using Machine Learning & Streamlit"
)
# =====================================================
# USER INPUT FORM
# =====================================================

left, right = st.columns(2)

# =====================================================
# LEFT CARD
# =====================================================

with left:

    st.markdown("""
    <div class="card">
    <div class="section-title">
    👤 Personal Details
    </div>
    """, unsafe_allow_html=True)

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=22
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    height = st.number_input(
        "Height (cm)",
        min_value=120.0,
        max_value=220.0,
        value=170.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0
    )

    goal = st.selectbox(
        "Fitness Goal",
        [
            "Weight Loss",
            "Muscle Gain",
            "Maintain Weight"
        ]
    )

    activity = st.selectbox(
        "Activity Level",
        [
            "Sedentary",
            "Light",
            "Moderate",
            "Active"
        ]
    )

    experience = st.selectbox(
        "Workout Experience",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RIGHT CARD
# =====================================================

with right:

    st.markdown("""
    <div class="card">
    <div class="section-title">
    ❤️ Health Details
    </div>
    """, unsafe_allow_html=True)

    diet_pref = st.selectbox(
    "Diet Preference",
    [
        "Vegetarian",
        "Non-Vegetarian"
    ]
)

    medical_options = ["No Medical Condition", "Diabetes", "Hypertension"]
    if gender == "Female":
        medical_options.append("PCOS")

    medical = st.selectbox(
        "Medical Condition",
        medical_options
    )

    sleep = st.slider(
        "Sleep Hours",
        4,
        10,
        7
    )

    water = st.slider(
        "Water Intake (Litres)",
        1,
        5,
        3
    )

    calories = st.number_input(
        "Daily Calories",
        min_value=1000,
        max_value=5000,
        value=2200
    )

    workout_days = st.slider(
        "Workout Days / Week",
        0,
        7,
        4
    )

    smoking = st.selectbox(
        "Smoking",
        [
            "Yes",
            "No"
        ]
    )

    alcohol = st.selectbox(
        "Alcohol",
        [
            "Never",
            "Occasionally",
            "Frequently"
        ]
    )

    stress = st.selectbox(
        "Stress Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    allergy = st.selectbox(
        "Food Allergy",
        [
            "None",
            "Nuts",
            "Dairy",
            "Gluten"
        ]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# BMI
# =====================================================

bmi = round(weight / ((height / 100) ** 2), 2)

st.write("")

# =====================================================
# PREDICT BUTTON
# =====================================================

predict = st.button(
    "🚀 Generate AI Recommendation",
    use_container_width=True
)

st.write("")
st.write("")

# =====================================================
# AI PREDICTION
# =====================================================

if predict:

    # =====================================================
    # CREATE INPUT DATAFRAME FOR THE PIPELINES
    # =====================================================

    user_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Height_cm": [height],
        "Weight_kg": [weight],
        "BMI": [bmi],
        "Goal": [goal],
        "ActivityLevel": [activity],
        "DietPreference": [diet_pref],
        "MedicalCondition": [medical],
        "WorkoutExperience": [experience],
        "SleepHours": [sleep],
        "WaterIntake_L": [water],
        "DailyCalories": [calories],
        "WorkoutDays": [workout_days],
        "Smoking": [smoking],
        "Alcohol": [alcohol],
        "StressLevel": [stress],
        "FoodAllergy": [allergy]
    })

    # =====================================================
    # MODEL PREDICTION (Pipelines loaded via joblib)
    # =====================================================

    predicted_diet = diet_pipeline.predict(user_data)[0]
    predicted_workout = workout_pipeline.predict(user_data)[0]

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    st.session_state["diet"] = predicted_diet

    st.session_state["workout"] = predicted_workout

    st.session_state["bmi"] = bmi

    st.session_state["status"] = (
        "Underweight"
        if bmi < 18.5 else
        "Normal"
        if bmi < 25 else
        "Overweight"
        if bmi < 30 else
        "Obese"
    )

    st.session_state["calories"] = calories

    st.session_state["protein"] = round(weight * 2.2)

    st.session_state["carbs"] = round(calories * 0.50 / 4)

    st.session_state["fat"] = round(calories * 0.25 / 9)

    st.session_state["water"] = water

    st.session_state["sleep"] = sleep

    st.session_state["goal"] = goal

    st.session_state["diet_pref"] = diet_pref

    st.session_state["medical"] = medical

    st.session_state["workout_days"] = workout_days

    st.session_state["smoking"] = smoking

    st.session_state["alcohol"] = alcohol

    st.session_state["stress"] = stress

    st.success("Recommendation Generated Successfully!")

# =====================================================
# RESULTS DASHBOARD
# =====================================================

if "diet" in st.session_state:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="main-title">
    🎉 Your Personalized Fitness Recommendation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sub-title">
    AI Generated Health & Fitness Report
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # DIET & WORKOUT CARDS
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="result-card">
            <h2>🥗 Recommended Diet</h2>
            <hr>
            <h3 style="color:#2563EB;">
            {st.session_state["diet"]}
            </h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="result-card">
            <h2>💪 Recommended Workout</h2>
            <hr>
            <h3 style="color:#2563EB;">
            {st.session_state["workout"]}
            </h3>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # =====================================
    # HEALTH SUMMARY
    # =====================================

    st.subheader("📊 Health Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "BMI",
            st.session_state["bmi"]
        )

    with c2:
        st.metric(
            "Health Status",
            st.session_state["status"]
        )

    with c3:
        st.metric(
            "Workout Days",
            st.session_state["workout_days"]
        )

    st.write("")

    # =====================================
    # NUTRITION DASHBOARD
    # =====================================

    st.subheader("🔥 Daily Nutrition")

    n1, n2, n3, n4 = st.columns(4)

    with n1:
        st.metric(
            "Calories",
            f'{st.session_state["calories"]} kcal'
        )

    with n2:
        st.metric(
            "Protein",
            f'{st.session_state["protein"]} g'
        )

    with n3:
        st.metric(
            "Carbohydrates",
            f'{st.session_state["carbs"]} g'
        )

    with n4:
        st.metric(
            "Fat",
            f'{st.session_state["fat"]} g'
        )

    st.write("")

    # =====================================
    # FITNESS SUMMARY CARD
    # =====================================

    st.markdown("""
    <div class="card">
    <h2 style="color:#2563EB;">
    📋 Fitness Summary
    </h2>
    """, unsafe_allow_html=True)

    st.write(f"""
### 👤 Profile

- **Goal:** {st.session_state["goal"]}

- **Diet Preference:** {st.session_state["diet_pref"]}

- **Medical Condition:** {st.session_state["medical"]}

- **Workout Days:** {st.session_state["workout_days"]} Days / Week

- **Daily Calories:** {st.session_state["calories"]} kcal

- **Water Intake:** {st.session_state["water"]} Litres

- **Sleep:** {st.session_state["sleep"]} Hours

- **BMI:** {st.session_state["bmi"]}

- **Health Status:** {st.session_state["status"]}
""")

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # =====================================================
    # SUGGESTED MEAL PLAN
    # =====================================================

    st.subheader("🍽 Suggested Meal Plan")

    dp = st.session_state.get("diet_pref", "Vegetarian")

    if st.session_state["goal"] == "Muscle Gain":
        if dp in ["Vegetarian", "Vegan"]:
            breakfast = "🥣 Oats + Milk + Almonds + Banana"
            lunch = "🍛 Paneer Curry + Brown Rice + Dal"
            snack = "🥜 Peanut Butter Toast + Seeds"
            dinner = "🍲 Tofu Stir-fry + Chapati + Salad"
            post = "🥤 Whey Protein / Soya Milk + Banana"
        else:
            breakfast = "🥣 Oats + Eggs + Banana"
            lunch = "🍗 Chicken + Rice + Vegetables"
            snack = "🥜 Peanut Butter Sandwich"
            dinner = "🥩 Chicken + Chapati + Salad"
            post = "🥤 Whey Protein + Banana"

    elif st.session_state["goal"] == "Weight Loss":
        if dp in ["Vegetarian", "Vegan"]:
            breakfast = "🥣 Oats + Apple"
            lunch = "🥗 Brown Rice + Sautéed Tofu / Dal"
            snack = "🥜 Almonds + Green Tea"
            dinner = "🥬 Vegetable Salad + Paneer"
            post = "🥛 Greek Yogurt"
        else:
            breakfast = "🥣 Oats + Egg Whites + Apple"
            lunch = "🥗 Brown Rice + Grilled Chicken"
            snack = "🥜 Almonds + Green Tea"
            dinner = "🥬 Vegetable Salad + Grilled Fish"
            post = "🥛 Greek Yogurt"

    else:
        if dp in ["Vegetarian", "Vegan"]:
            breakfast = "🥣 Oats + Milk"
            lunch = "🍚 Rice + Dal + Vegetables"
            snack = "🍎 Fruits + Nuts"
            dinner = "🍛 Chapati + Mixed Veg Curry"
            post = "🥛 Milk"
        else:
            breakfast = "🥣 Oats + Eggs / Milk"
            lunch = "🍚 Rice + Dal + Chicken/Fish"
            snack = "🍎 Fruits + Nuts"
            dinner = "🍛 Chapati + Egg/Chicken Curry"
            post = "🥛 Milk"

    # Make adjustments for vegan specifically if selected
    if dp == "Vegan":
        breakfast = breakfast.replace("Milk", "Almond Milk").replace("Greek Yogurt", "Vegan Yogurt")
        lunch = lunch.replace("Paneer", "Tofu")
        dinner = dinner.replace("Paneer", "Tofu")
        post = post.replace("Whey Protein", "Soy Protein").replace("Greek Yogurt", "Vegan Yogurt").replace("Milk", "Soy Milk")

    meal1, meal2 = st.columns(2)

    with meal1:

        st.info(f"**Breakfast**\n\n{breakfast}")

        st.info(f"**Lunch**\n\n{lunch}")

        st.info(f"**Snack**\n\n{snack}")

    with meal2:

        st.info(f"**Dinner**\n\n{dinner}")

        st.success(f"**Post Workout**\n\n{post}")

    st.write("")

    # =====================================================
    # SUGGESTED WORKOUT SPLIT
    # =====================================================

    st.subheader("🏋️ Suggested Workout Split")

    wc = st.session_state["workout"]
    wd = st.session_state["workout_days"]

    if wc in WORKOUT_PLANS:
        plan = WORKOUT_PLANS[wc]
        schedule = generate_schedule(plan, wd)

        cols = st.columns(7)
        for idx, day in enumerate(['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']):
            with cols[idx]:
                desc = schedule[day]
                if "Rest" in desc:
                    st.success(f"**{day}**\n\n{desc}")
                else:
                    st.info(f"**{day}**\n\n{desc}")
    else:
        st.warning("Workout split schedule not available for this classification.")

    st.write("")

    # =====================================================
    # WATER INTAKE
    # =====================================================

    st.subheader("💧 Water Intake")

    progress = min(st.session_state["water"] / 5, 1.0)

    st.progress(progress)

    st.success(
        f"Daily Water Intake : {st.session_state['water']} Litres"
    )

    st.write("")

    # =====================================================
    # SLEEP ANALYSIS
    # =====================================================

    st.subheader("😴 Sleep Analysis")

    if st.session_state["sleep"] < 6:

        st.warning(
            "⚠ You should sleep at least 7-8 hours for proper muscle recovery."
        )

    elif st.session_state["sleep"] <= 8:

        st.success(
            "✅ Excellent! Your sleep schedule is healthy."
        )

    else:

        st.info(
            "💤 Sleeping too much may affect your daily routine."
        )

    st.write("")

    # =====================================================
    # HEALTH TIPS
    # =====================================================

    st.subheader("❤️ Personalized Health Tips")

    tips = []

    if st.session_state["bmi"] > 25:
        tips.append("Reduce sugar and processed foods.")

    if st.session_state["water"] < 3:
        tips.append("Increase your daily water intake.")

    if st.session_state["sleep"] < 7:
        tips.append("Maintain a consistent sleep schedule.")

    if st.session_state["smoking"] == "Yes":
        tips.append("Avoid smoking to improve overall fitness.")

    if st.session_state["alcohol"] == "Frequently":
        tips.append("Reduce alcohol consumption.")

    if st.session_state["stress"] == "High":
        tips.append("Practice meditation or yoga daily.")

    if len(tips) == 0:

        st.success(
            "🎉 Great job! Your lifestyle habits look healthy."
        )

    else:

        for tip in tips:

            st.info("✅ " + tip)

    st.write("")

    # =====================================================
    # DAILY SUMMARY
    # =====================================================

    st.subheader("📋 Personalized Fitness Report")

    st.code(f"""
========== PERSONALIZED FITNESS RECOMMENDATION ==========

Recommended Diet         : {st.session_state["diet"]}

Recommended Workout      : {st.session_state["workout"]}

Daily Calories           : {st.session_state["calories"]} kcal

Protein                  : {st.session_state["protein"]} g/day

Carbohydrates            : {st.session_state["carbs"]} g/day

Fat                      : {st.session_state["fat"]} g/day

Workout Frequency        : {st.session_state["workout_days"]} Days/Week

Water Intake             : {st.session_state["water"]} Litres

Recommended Sleep        : {st.session_state["sleep"]} Hours

=========================================================
""")

    st.write("")

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("---")

    st.markdown("""
    <div style="background:white;
                padding:20px;
                border-radius:15px;
                text-align:center;
                box-shadow:0px 5px 20px rgba(0,0,0,.08);">

        <h3 style="color:#2563EB;">
        🤖 AI Diet & Workout Recommendation System
        </h3>

        <p style="color:#6B7280;">
        Developed using Machine Learning,
        Random Forest and Streamlit
        </p>

        <p style="color:#9CA3AF;">
        © 2026 All Rights Reserved
        </p>

    </div>
    """, unsafe_allow_html=True)

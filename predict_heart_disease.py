import joblib
import pandas as pd
import pyinputplus as pyip

ml_model = joblib.load("./actual_model/actual_model.joblib")

age_categories = [
    '18-24',
    '25-29',
    '30-34',
    '35-39',
    '40-44',
    '45-49',
    '50-54',
    '55-59',
    '60-64',
    '65-69',
    '70-74',
    '75-79',
    '80 or older'
]
len_age_categories = len(age_categories)
# The list below will end up having values from 0 to 1, with the same distance between each other [0.0, 0.083, 0.166...]
age_categories_as_numbers = [(1 / (len_age_categories - 1)) * idx for idx in range(len_age_categories)]

age_categories_to_numbers = {key: value for key, value in zip(age_categories, age_categories_as_numbers)}

age_category = pyip.inputMenu(prompt="Enter the number corresponding to your age category:\n", choices=age_categories, numbered=True)
age_category = age_categories_to_numbers[age_category]

difficulty_walking = pyip.inputYesNo(prompt="Do you have serious difficulty walking or climbing stairs?\n")
if difficulty_walking == "no":
    difficulty_walking = 0
else:
    difficulty_walking = 1

diabetic = pyip.inputYesNo(prompt="Have you ever been diagnosed with diabetes?\n")
if diabetic == "no":
    diabetic = 0
else:
    diabetic = 1

stroke = pyip.inputYesNo(prompt="Have you ever had a stroke?\n")
if stroke == "no":
    stroke = 0
else:
    stroke = 1

general_health_markers = ["Poor", "Fair", "Excellent", "Good", "Very good"]
general_health_markers_to_numbers = {
    "Poor": 0,
    "Fair": 0.25,
    "Excellent": 0.5,
    "Good": 0.75,
    "Very good": 1
}
general_health = pyip.inputMenu(prompt="Your health in general is...\n", choices=general_health_markers, numbered=True)
general_health = general_health_markers_to_numbers[general_health]

model_input = pd.DataFrame({
    "AgeCategory": [age_category],
    "DiffWalking": [difficulty_walking],
    "Diabetic": [diabetic],
    "Stroke": [stroke],
    "GenHealth": [general_health]
})

prediction = ml_model.predict(model_input)[0]

if prediction == 0:
    print("You are not at risk of heart disease!")
else:
    print("You are at risk of heart disease!")

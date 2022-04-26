"""
This program is used to generate a DecisionTreeClassifier model that predicts if someone is at risk of heart disease.
The model is saved in the temp_model folder, and it can later be manually placed in the actual_model folder.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib


def main():
    df = pd.read_csv("heart_2020_cleaned.csv")

    df.head()

    # Replacing all strings in dataframe with numbers
    df["HeartDisease"] = df["HeartDisease"].replace(["No", "Yes"], [0, 1])
    df["Smoking"] = df["Smoking"].replace(["No", "Yes"], [0, 1])
    df["AlcoholDrinking"] = df["AlcoholDrinking"].replace(["No", "Yes"], [0, 1])
    df["Stroke"] = df["Stroke"].replace(["No", "Yes"], [0, 1])
    df["DiffWalking"] = df["DiffWalking"].replace(["No", "Yes"], [0, 1])
    df["PhysicalActivity"] = df["PhysicalActivity"].replace(["No", "Yes"], [0, 1])
    df["Asthma"] = df["Asthma"].replace(["No", "Yes"], [0, 1])
    df["KidneyDisease"] = df["KidneyDisease"].replace(["No", "Yes"], [0, 1])
    df["SkinCancer"] = df["SkinCancer"].replace(["No", "Yes"], [0, 1])
    df["Diabetic"] = df["Diabetic"].replace(["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"], [0, 0, 1, 1])
    df["Sex"] = df["Sex"].replace(["Female", "Male"], [0, 1])
    races = ["Other", "White", "American Indian/Alaskan Native", "Hispanic", "Asian", "Black"]
    races_as_numbers = [0, 0.2, 0.4, 0.6, 0.8, 1]
    df["Race"] = df["Race"].replace(races, races_as_numbers)
    general_health_markers = ["Poor", "Fair", "Excellent", "Good", "Very good"]
    general_health_markers_as_numbers = [0, 0.25, 0.5, 0.75, 1]
    df["GenHealth"] = df["GenHealth"].replace(general_health_markers, general_health_markers_as_numbers)
    age_categories = list(df["AgeCategory"].value_counts().index)
    age_categories.sort()
    len_age_categories = len(age_categories)
    # The list below will end up having values from 0 to 1, with the same distance between each other
    age_categories_as_numbers = [(1 / (len_age_categories - 1)) * idx for idx in range(len_age_categories)]
    df["AgeCategory"] = df["AgeCategory"].replace(age_categories, age_categories_as_numbers)
    df["BMI"] = df["BMI"].astype("int")

    new_df = undersample_dataset(df, "HeartDisease", 90)
    X = new_df[["AgeCategory", "DiffWalking", "Diabetic", "Stroke", "GenHealth"]]
    y = new_df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    print("acc:", acc)
    print("prec:", prec)
    print("recall:", recall)
    print("f1:", f1)

    joblib.dump(model, "./temp_model/temp_model.joblib")


def undersample_dataset(dataframe, y_column, percentage_removed):
    """
    Returns dataframe with some of the rows where y_column has the value 0 removed
    The amount of rows removed is determined by percentage_removed
    """

    rows_with_target_variable_0 = dataframe[dataframe[y_column] == 0]
    len_rows_with_target_variable_0 = len(rows_with_target_variable_0)

    new_dataframe = rows_with_target_variable_0[int(len_rows_with_target_variable_0 * (percentage_removed / 100)):]
    new_dataframe = pd.concat([dataframe[dataframe[y_column] == 1], new_dataframe])

    return new_dataframe


if __name__ == "__main__":
    main()

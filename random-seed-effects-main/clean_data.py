import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from static import *
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def clean_data(data_set_name):
    # the path of the original data
    base_path_original = f"./{DATA_FOLDER}/{data_set_name}/{ORIGINAL_FOLDER}"

    # load the data into a dataframe
    if data_set_name == "WeatherData":
        data = pd.read_csv(f"{base_path_original}/weather.csv")

        # Initialize the LabelEncoder
        le_summary = LabelEncoder()
        le_daily_summary = LabelEncoder()
        le_date = LabelEncoder()

        # Fit and transform the categorical variables
        data['Daily_Summary'] = le_daily_summary.fit_transform(data['Daily_Summary'])
        data['Summary'] = le_summary.fit_transform(data['Summary'])
        data['Formatted_Date'] = le_date.fit_transform(data['Formatted_Date'])

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Select numerical columns to scale
        numerical_cols = ['Temperature', 'Apparent_Temperature', 'Humidity', 'Wind_Speed', 'Wind_Bearing', 'Visibility', 'Pressure']

        # Fit and transform the numerical features
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        data.rename(columns={
            'Temperature': 'X',
            'Apparent_Temperature': 'y'

        }, inplace=True)
    elif data_set_name == "WhoData":
        data = pd.read_csv(f"{base_path_original}/LifeExpData.csv")

        data['CountryYear'] = data['Country'].astype(str) + '_' + data['Year'].astype(str)
        data = data.drop(columns=['Country', 'Year'])

        # Initialize the LabelEncoder
        le_country_year = LabelEncoder()
        le_status = LabelEncoder()

        # Fit and transform the categorical variables
        data['CountryYear'] = le_country_year.fit_transform(data['CountryYear'])
        data['Status'] = le_status.fit_transform(data['Status'])
        data.rename(columns={
            'CountryYear': 'X',
            'Status': 'y'
        }, inplace=True)
        # Initialize the StandardScaler
        scaler = StandardScaler()

    elif data_set_name == "CancerData2":
        data = pd.read_csv(f"{base_path_original}/wdbc.data")

        # Initialize the LabelEncoder
        le_diagnosis = LabelEncoder()
        le_user = LabelEncoder()

        # Fit and transform the categorical variables
        data['Diagnosis'] = le_diagnosis.fit_transform(data['Diagnosis'])
        data['user'] = le_user.fit_transform(data['user'])
        data.rename(columns={
            'Diagnosis': 'X',
            'user': 'y'
        }, inplace=True)
    elif data_set_name == "AdultData":
        data = pd.read_csv(f"{base_path_original}/adult.data")

        # Add an ID column starting from 1
        data['id'] = range(1, len(data) + 1)

        # Initialize the LabelEncoder
        le_workclass = LabelEncoder()
        le_marital = LabelEncoder()
        le_occupation = LabelEncoder()
        le_relationship = LabelEncoder()
        le_race = LabelEncoder()
        le_sex = LabelEncoder()
        le_country = LabelEncoder()
        le_salary = LabelEncoder()

        # Fit and transform the categorical variables
        data['workclass'] = le_workclass.fit_transform(data['workclass'])
        data['marital'] = le_marital.fit_transform(data['marital'])
        data['occupation'] = le_occupation.fit_transform(data['occupation'])
        data['relationship'] = le_relationship.fit_transform(data['relationship'])
        data['race'] = le_race.fit_transform(data['race'])
        data['sex'] = le_sex.fit_transform(data['sex'])
        data['country'] = le_country.fit_transform(data['country'])
        data['salary'] = le_salary.fit_transform(data['salary'])

        data = data.drop(columns=["education"])
        data.rename(columns={
            'occupation': 'X',
            'salary': 'y'
        }, inplace=True)
    elif data_set_name == "SpamData":
        data = pd.read_csv(f"{base_path_original}/spambase.csv")

        # Add an ID column starting from 1
        data['id'] = range(1, len(data) + 1)

        # Initialize the StandardScaler
        scaler = StandardScaler()

        numerical_cols = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
                          "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
                          "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report",
                          "word_freq_addresses", "word_freq_free", "word_freq_business", "word_freq_email",
                          "word_freq_you", "word_freq_credit", "word_freq_your", "word_freq_font", "word_freq_000",
                          "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
                          "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data",
                          "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts",
                          "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original",
                          "word_freq_project", "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
                          "char_freq_%3B", "char_freq_%28", "char_freq_%5B", "char_freq_%21", "char_freq_%24",
                          "char_freq_%23", "capital_run_length_average", "capital_run_length_longest",
                          "capital_run_length_total", "class"]

        # Fit and transform the numerical features
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        data.rename(columns={
            'word_freq_all': 'X',
            'capital_run_length_total': 'y',
        }, inplace=True)
    else:
        raise ValueError(f"Unknown data set name {data_set_name}.")

    # remove duplicates
    # data.drop_duplicates(inplace=True)

    # map user and item to integers if they exist
    if "X" in data.columns and "Y" in data.columns:
        for col in ["X", "Y"]:
            unique_ids = {key: value for value, key in enumerate(data[col].unique())}
            data[col].update(data[col].map(unique_ids))
        print("Dropped duplicates and mapped user and item to integers.")
    else:
        print("Dropped duplicates.")

    print("Dropped duplicates and mapped user and item to integers.")

    # write data to file
    base_path_cleaned = f"./{DATA_FOLDER}/{data_set_name}/{CLEAN_FOLDER}"
    Path(base_path_cleaned).mkdir(exist_ok=True)
    data.to_csv(f"{base_path_cleaned}/{CLEAN_FILE}", index=False)
    print(f"Written cleaned data set to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects clean data!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    args = parser.parse_args()

    print("Pruning original with arguments: ", args.__dict__)
    clean_data(args.data_set_name)

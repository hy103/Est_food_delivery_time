import pandas as pd
import pickle
import os
from data_modelling import data_preprocessing
import warnings
warnings.filterwarnings("ignore")



def main():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_PATH,"../models/gbm.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_PATH,"../models/onehot_encoder"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(BASE_PATH,"../models/scaler"), "rb") as f:
        scaler = pickle.load(f)

    delivery_person_age = int(input("Enter Delivery person's age: "))
    delivery_person_ratings = float(input("Enter Delivery person's ratings (e.g., 4.5): "))
    weather_conditions = input("Enter Weather conditions (e.g., 'conditions Cloudy'): ")
    road_traffic_density = input("Enter Road traffic density (e.g., 'Medium '): ")
    vehicle_condition = int(input("Enter Vehicle condition (integer): "))
    type_of_order = input("Enter Type of order (e.g., 'Buffet '): ")
    type_of_vehicle = input("Enter Type of vehicle (e.g., 'motorcycle '): ")
    multiple_deliveries = int(input("Enter Multiple deliveries (e.g., 1): "))
    festival = input("Is it a festival? ('No ' or 'Yes '): ")
    city = input("Enter City (e.g., 'Urban '): ")
    distance_rest_del_loc = float(input("Enter distance from restaurant to delivery location (in miles): "))
    hour = int(input("Enter hour of order (24-hour format, e.g., 17): "))
    day = int(input("Enter day of the month (e.g., 11): "))
    weekday_name = input("Enter weekday name (e.g., 'Friday'): ")
    type_of_meal = input("Enter type of meal (e.g., 'Dinner'): ")

    # Create the DataFrame from the user inputs
    single_row = pd.DataFrame({
        'Delivery_person_Age': [delivery_person_age],
        'Delivery_person_Ratings': [delivery_person_ratings],
        'Weatherconditions': [weather_conditions],
        'Road_traffic_density': [road_traffic_density],
        'Vehicle_condition': [vehicle_condition],
        'Type_of_order': [type_of_order],
        'Type_of_vehicle': [type_of_vehicle],
        'multiple_deliveries': [multiple_deliveries],
        'Festival': [festival],
        'City': [city],
        'distance_rest_del_loc': [distance_rest_del_loc],
        'hour': [hour],
        'day': [day],
        'weekday_name': [weekday_name],
        'TypeOfMeal': [type_of_meal]
    })
    
    # Convert single Series row to DataFrame
    single_row_df = single_row
    single_row_preprocessed = data_preprocessing(single_row_df, train= False, encoder= encoder, scaler = scaler)

    prediction = model.predict(single_row_preprocessed)
    return prediction


if __name__ == '__main__':
    predicted_value = main()
    print(f"The Estimated time to deliver order is : {round(predicted_value[0],0)} mins")
import pandas as pd 
import re
import datetime
from dateutil.relativedelta import relativedelta, MO, TH 
import geopy.distance
import os

def convert_string_to_numeric(x):
    if x is None or x == "":
        return pd.NA
    try:
        return float(x)
    except ValueError :
        return pd.NA

def convert_string_to_numeric_columns(df):
    numeric_columns = ['Delivery_person_Age',
        'Delivery_person_Ratings', 'Restaurant_latitude',
        'Restaurant_longitude', 'Delivery_location_latitude',
        'Delivery_location_longitude', 'Vehicle_condition', 
        'multiple_deliveries', 'Time_taken(min)']

    for col in numeric_columns:
        if col == "Time_taken(min)":
            df[col] = df[col].apply(lambda x : convert_string_to_numeric(re.search(r'\d+', x).group()))
        else:
            df[col] = df[col].apply(lambda x : convert_string_to_numeric(x))

    return df


def check_null_values_and_format(df, col_name):
    print(f"The column {col_name} is of data type {df[col_name].dtype}")
    if df[col_name].isnull().sum() == 0:
        return print("No Null values are there")
    else :
        return print("Null values")
    
def remove_records_with_no_ID(df, col_name):
    df = df.dropna(subset = [col_name])
    return df


def replace_string_na_with_NA(df):
    return df.replace('NaN ' , pd.NA)

#Check latitude
def is_latitude(value):
    return -90<=value<=90

#Check longitude
def is_longitude(value):
    return -180<=value<=180

### Replace 0 or other non latitude and longitude values
def replace_non_lat_lang_with_NA(df, geography_col):
    if geography_col.endswith("latitude"):
        df[geography_col] = df[geography_col].apply(lambda x : pd.NA if x is not None 
                                                    and not is_latitude(x) else x)
    elif geography_col.endswith("longitude"):
        df[geography_col] = df[geography_col].apply(lambda x : pd.NA if x is not None 
                                                    and not is_longitude(x) else x)
    else:
        pass
    return df

def lat_long_preprocessing(df):
    geogrpahy_columns = []
    for col_name in df.columns:
        if col_name.endswith("latitude") or col_name.endswith("longitude"):
            geogrpahy_columns.append(col_name)

    for col in geogrpahy_columns:
        replace_non_lat_lang_with_NA(df, col)

    return df


## Converting all date formats into a single format
def convert_into_ddmmyyyy_format(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce', dayfirst= True)
    df["date_column_ddmmyyy"] = df[date_column].dt.strftime('%d%m%Y')
    return df



def replace_missing_weather_traffic(df):
    df["Weatherconditions"] = df["Weatherconditions"].replace('conditions NaN', pd.NA)
    df["Road_traffic_density"] = df["Road_traffic_density"].replace('conditions NaN', pd.NA)
    return df


## Calculating % of missing values in each column
def return_missing_values(df):
    missing_values_df = pd.DataFrame(columns=["Column", "MissingPercentage"])
    for index, col_name in enumerate(df.columns):
        num_missing_values = df[col_name].isnull().sum()
        missing_values_df = pd.concat([missing_values_df, 
                                    pd.DataFrame({"Column": col_name , 
                                                    "MissingPercentage" :[num_missing_values/df.shape[0]*100]})])

    missing_values_df.reset_index(drop=True, inplace=True)
    return missing_values_df



def fill_delivery_missing_values(df):
    #Fill Delivery_person_Age with the median ratings
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(df["Delivery_person_Age"].median())

    #Fill Delivery_person_Ratings with the median ratings
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(df["Delivery_person_Ratings"].median())

    #Fill Multiple deliveries with the most repeating time
    most_frequent_del = df["multiple_deliveries"].mode()[0] 
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(most_frequent_del)
    return df

# Federal Holidays
HOLIDAYS = {
    "New Year's Day": lambda year: datetime.date(year, 1, 1),
    "Martin Luther King's Birthday": lambda year: datetime.date(year, 1, 1) + relativedelta(weekday=MO(+3)),
    "Washington's Birthday": lambda year: datetime.date(year, 2, 1) + relativedelta(weekday=MO(+3)),
    "Memorial Day": lambda year: datetime.date(year, 5, 31) + relativedelta(weekday=MO(-1)),
    "Juneteenth National Independence Day": lambda year: datetime.date(year, 6, 19),
    "Independence Day": lambda year: datetime.date(year, 7, 4),
    "Labor Day": lambda year: datetime.date(year, 9, 1) + relativedelta(weekday=MO(+1)),
    "Columbus Day": lambda year: datetime.date(year, 10, 1) + relativedelta(weekday=MO(+2)),
    "Veterans' Day": lambda year: datetime.date(year, 11, 11),
    "Thanksgiving Day": lambda year: datetime.date(year, 11, 1) + relativedelta(weekday=TH(+4)),
    "Christmas Day": lambda year: datetime.date(year, 12, 25),
}

# Function to check if a given date is a public holiday
def is_public_holiday(date_to_check):
    year = date_to_check.year
    for holiday_name, holiday_func in HOLIDAYS.items():
        if holiday_func(year) == date_to_check:
            return 'Yes '
    return 'No '


def convert_to_date(date):
    if isinstance(date, str):
        return datetime.datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, pd.Timestamp):
        return date.date()
    else:
        return None
    

# Update the "Festival" column only for values that are not 'No ' or 'Yes '
def fill_missing_festivals(df):
    df.loc[(df["Festival"] != 'No ') & 
                (df["Festival"] != 'Yes '), "Festival"] = df.loc[(df["Festival"] != 'No ') & 
                                                                 (df["Festival"] != 'Yes '), "Order_Date"].apply(lambda x: is_public_holiday(convert_to_date(x)))

    return df

## Replacing NA with most frequent Catgory of city
def fill_missing_city(df):
    most_frequent_city_category = df["City"].mode()[0] 
    df["City"] = df["City"].fillna(most_frequent_city_category)
    return df


def calculate_dist_res_del_locatio(x):
    res_lat_long = (abs(x['Restaurant_latitude']), abs(x['Restaurant_longitude']))
    del_lat_long = (abs(x['Delivery_location_latitude']), abs(x['Delivery_location_longitude']))

    return geopy.distance.geodesic(res_lat_long, del_lat_long).miles


def calcualate_time_diff(x1, x2 ):
    return ((pd.to_datetime(x1) -
            pd.to_datetime(x2)).total_seconds())/60.0


def date_features(df):
    df['date_column_str'] = df['date_column_ddmmyyy'].astype(str)
    df['date_column'] = pd.to_datetime(df['date_column_str'], format='%d%m%Y')
    df['Time_Orderd_str'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S').dt.time

    df['hour'] = df['Time_Orderd_str'].apply(lambda x: x.hour)
    df['day'] = df['date_column'].dt.day
    df['month'] = df['date_column'].dt.month
    df['year'] = df['date_column'].dt.year
    df['weekday_name'] = df['date_column'].dt.day_name()

    # Optional: Get the weekday as an integer (0=Monday, 6=Sunday)
    df['weekday_int'] = df['date_column'].dt.weekday
    df['TypeOfMeal'] = df['hour'].apply(lambda value: "Breakfast" if value in [8, 9, 10, 11] else "Lunch" if value in [12, 13, 14, 15, 16] else "Dinner")
    return df


def res_features(df):
    df.rename({'Delivery_person_ID' : 'Restaurant_ID'}, axis =1, inplace = True)
    df["distance_rest_del_loc"] = df.apply(lambda x : calculate_dist_res_del_locatio(x), axis =1)

    df["Time_diff_res_picked"] = df.apply(lambda x : 
                                        calcualate_time_diff(x["Time_Order_picked"], x["Time_Orderd"]), axis = 1)
    df['Time_diff_res_picked'] = df['Time_diff_res_picked'].apply(lambda x: x + 24 * 60 if x < 0 else x)
    return df

def preprocessed_data(df):
    df = convert_string_to_numeric_columns(df)
    df = remove_records_with_no_ID(df, "ID")
    df = replace_string_na_with_NA(df)
    df = lat_long_preprocessing(df)
    df = convert_into_ddmmyyyy_format(df, "Order_Date")
    df = replace_missing_weather_traffic(df)
    df = df.dropna(subset=["Time_Orderd"])
    df = fill_delivery_missing_values(df)
    df = fill_missing_festivals(df)
    df = fill_missing_city(df)
    df = res_features(df)
    df = date_features(df)
    df = df.drop(['ID', 'Restaurant_ID', 'Restaurant_latitude', 'Restaurant_longitude',
       'Delivery_location_latitude', 'Delivery_location_longitude',
       'Order_Date', 'Time_Orderd', 'Time_Order_picked',
       'month', 'year', 'date_column_ddmmyyy', 'Time_diff_res_picked','date_column_str', 'date_column',
       'Time_Orderd_str','weekday_int'], axis =1)

    return df


if __name__ == '__main__':
    ## Uncomment below lines if you want to test data preprocessing script alone
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #data_path = os.path.join(BASE_DIR, "../data/uber-eats-deliveries.csv")
    #df = pd.read_csv(data_path)
    #preproecess_data = preprocessed_data(df)
    pass


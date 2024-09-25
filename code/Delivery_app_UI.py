import streamlit as st
import folium
from streamlit_folium import st_folium
import geopy.distance
import pandas as pd
import random
from datetime import datetime
import pickle
import os
from data_modelling import data_preprocessing  # Assuming this is your custom preprocessing function

st.markdown("# Foodie 🍔")
st.write("Your convenient solution for ordering delicious meals from local restaurants, delivered right to your door!")

# Function to calculate distance in miles
def calculate_distance(restaurant_location, delivery_location):
    """Calculate distance in miles between restaurant and delivery locations."""
    return geopy.distance.geodesic(restaurant_location, delivery_location).miles

# Sample restaurant menu
menu = {
    "Burgers": ["Cheeseburger", "Bacon Burger", "Veggie Burger"],
    "Pizzas": ["Margherita", "Pepperoni", "BBQ Chicken"],
    "Salads": ["Caesar Salad", "Greek Salad", "Garden Salad"],
    "Desserts": ["Chocolate Cake", "Ice Cream", "Fruit Salad"],
    "Drinks": ["Coke", "Lemonade", "Water"]
}

# Initialize session state for locations if not already done
if 'restaurant_location' not in st.session_state:
    st.session_state.restaurant_location = None
if 'delivery_location' not in st.session_state:
    st.session_state.delivery_location = None
if 'delivery_selected' not in st.session_state:
    st.session_state.delivery_selected = False
if 'predicted_time' not in st.session_state:
    st.session_state.predicted_time = None

# Check if both locations have been selected
if st.session_state.delivery_selected and st.session_state.restaurant_location and st.session_state.delivery_location:
    # Calculate the distance (but don't print it)
    distance = calculate_distance(st.session_state.restaurant_location, st.session_state.delivery_location)

    # Prompt user to select from the new dropdown options
    st.write("### Customize Your Delivery")

    # Dropdown for Delivery Person Ratings
    delivery_person_ratings = st.selectbox(
        "Select Delivery Person Ratings (5.0-1.0):",
        [round(x * 0.1, 1) for x in range(50, 9, -1)]  # Generating values from 5.0 to 1.0
    )

    # Dropdown for Type of Order
    type_of_order = st.selectbox(
        "Select Type of Order:",
        ['Buffet ', 'Meal ', 'Snack ']
    )

    # Dropdown for Multiple Deliveries
    multiple_deliveries = st.selectbox(
        "Select Number of Multiple Deliveries:",
        [0, 1, 2, 3]
    )

    # Dropdown for Type of Meal
    type_of_meal = st.selectbox(
        "Select Type of Meal:",
        ['Breakfast', 'Lunch', 'Dinner']
    )

    # Once the locations are selected, generate the single row data
    def generate_single_row():
        # Get current date and time
        current_time = datetime.now()
        
        # Extracting hour, day, and weekday name
        hour = current_time.hour  # Current hour in 24-hour format
        day = current_time.day    # Day of the month
        weekday_name = current_time.strftime("%A")  # Full name of the weekday

        # Other features (randomized)
        delivery_person_age = random.randint(20, 30)  # Random age between 20 and 30
        weather_conditions = random.choice(['conditions Sandstorms', 'conditions Fog', 'conditions Stormy', 'conditions Cloudy', 'conditions Sunny', 'conditions Windy'])  # Random weather condition
        road_traffic_density = random.choice(['Medium ', 'High ', 'Low '])  # Random road traffic density
        vehicle_condition = random.choice([3, 2, 1, 0])  # Random vehicle condition
        type_of_vehicle = random.choice(['electric_scooter ', 'motorcycle ', 'scooter '])  # Random type of vehicle
        festival = random.choice(['No ', 'Yes '])  # Random festival status
        city = random.choice(['Metropolitian ', 'Urban ', 'Semi-Urban '])  # Random city type

        # Creating the DataFrame
        single_row = pd.DataFrame({
            'Delivery_person_Age': [delivery_person_age],
            'Delivery_person_Ratings': [delivery_person_ratings],  # User-selected
            'Weatherconditions': [weather_conditions],
            'Road_traffic_density': [road_traffic_density],
            'Vehicle_condition': [vehicle_condition],
            'Type_of_order': [type_of_order],  # User-selected
            'Type_of_vehicle': [type_of_vehicle],
            'multiple_deliveries': [multiple_deliveries],  # User-selected
            'Festival': [festival],
            'City': [city],
            'distance_rest_del_loc': [distance],  # Using calculated distance
            'hour': [hour],
            'day': [day],
            'weekday_name': [weekday_name],
            'TypeOfMeal': [type_of_meal]  # User-selected
        })

        return single_row

    # Generate and display the single row as text
    single_row_df = generate_single_row()

    # Load the model, encoder, and scaler
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    
    # Load the pre-trained model
    with open(os.path.join(BASE_PATH, "../models/gbm.pkl"), "rb") as f:
        model = pickle.load(f)

    # Load the one-hot encoder
    with open(os.path.join(BASE_PATH, "../models/onehot_encoder"), "rb") as f:
        encoder = pickle.load(f)

    # Load the scaler
    with open(os.path.join(BASE_PATH, "../models/scaler"), "rb") as f:
        scaler = pickle.load(f)

    # Create a button to predict delivery time
    if st.button("Predict Delivery Time"):
        # Preprocess the single row data
        single_row_preprocessed = data_preprocessing(single_row_df, train=False, encoder=encoder, scaler=scaler)

        # Make a prediction using the pre-trained model
        prediction = model.predict(single_row_preprocessed)
        st.session_state.predicted_time = prediction[0]  # Save the prediction to session state

    # Display the prediction result creatively
    if st.session_state.predicted_time is not None:
        st.markdown("### Estimated Delivery Time")
        # Creative styling for the output
        st.markdown(
            f"""
            <div style="background-color: #FF6347; padding: 20px; border-radius: 10px; text-align: center;">
                <h1 style="color: white; font-size: 50px;">{st.session_state.predicted_time:.2f} minutes</h1>
                <p style="color: white;">Your estimated delivery time is shown above!</p>
            </div>
            """, unsafe_allow_html=True
        )

else:
    # Create a dropdown menu for users to select a category
    category = st.selectbox("Select a category:", list(menu.keys()))

    # Display the items in the selected category
    if category:
        selected_items = menu[category]
        selected_item = st.selectbox("Select an item:", selected_items)

        # Display selected item and hide the menu
        if selected_item:
            st.write(f"You selected: **{selected_item}** from the **{category}** menu.")
            
            # Proceed to restaurant location selection
            st.info("Now, please select your restaurant location on the map.")

            # Set initial location to Tempe, AZ
            initial_lat, initial_lon = 33.4261, -111.9398

            # Create a Folium map centered at the initial location
            m = folium.Map(location=[initial_lat, initial_lon], zoom_start=12)

            # Add click event listener to the map
            m.add_child(folium.LatLngPopup())

            # Display the map using streamlit-folium
            st_data = st_folium(m, width=700, height=500)

            # Step 1: Capture Restaurant Location
            if st_data['last_clicked'] is not None:
                if st.session_state.restaurant_location is None:
                    # Capture the restaurant location
                    st.session_state.restaurant_location = (st_data['last_clicked']['lat'], st_data['last_clicked']['lng'])
                    st.info("Now, please select your delivery location on the map.")
                else:
                    # Step 2: Capture Delivery Location
                    st.session_state.delivery_location = (st_data['last_clicked']['lat'], st_data['last_clicked']['lng'])
                    
                    # Ensure delivery location is different from restaurant location
                    if st.session_state.restaurant_location == st.session_state.delivery_location:
                        st.warning("Delivery location cannot be the same as restaurant location. Please select a different location.")
                        st.session_state.delivery_location = None  # Reset delivery location
                    else:
                        st.session_state.delivery_selected = True  # Mark delivery location as selected

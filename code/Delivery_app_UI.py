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
    distance = calculate_distance(st.session_state.restaurant_location, st.session_state.delivery_location)
    st.write("### Customize Your Delivery")
    delivery_person_ratings = st.selectbox(
        "Select Delivery Person Ratings (5.0-1.0):",
        [round(x * 0.1, 1) for x in range(50, 9, -1)] 
    )

    type_of_order = st.selectbox(
        "Select Type of Order:",
        ['Buffet ', 'Meal ', 'Snack ']
    )

    multiple_deliveries = st.selectbox(
        "Select Number of Multiple Deliveries:",
        [0, 1, 2, 3]
    )

    type_of_meal = st.selectbox(
        "Select Type of Meal:",
        ['Breakfast', 'Lunch', 'Dinner']
    )

    def generate_single_row():
        current_time = datetime.now()
        hour = current_time.hour 
        day = current_time.day    
        weekday_name = current_time.strftime("%A")  

        # Other features
        delivery_person_age = random.randint(20, 30)  
        weather_conditions = random.choice(['conditions Sandstorms', 'conditions Fog', 'conditions Stormy', 'conditions Cloudy', 'conditions Sunny', 'conditions Windy']) 
        road_traffic_density = random.choice(['Medium ', 'High ', 'Low ']) 
        vehicle_condition = random.choice([3, 2, 1, 0]) 
        type_of_vehicle = random.choice(['electric_scooter ', 'motorcycle ', 'scooter ']) 
        festival = random.choice(['No ', 'Yes ']) 
        city = random.choice(['Metropolitian ', 'Urban ', 'Semi-Urban '])  

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
            'distance_rest_del_loc': [distance], 
            'hour': [hour],
            'day': [day],
            'weekday_name': [weekday_name],
            'TypeOfMeal': [type_of_meal]  
        })

        return single_row
    
    single_row_df = generate_single_row()
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(BASE_PATH, "../models/gbm.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_PATH, "../models/onehot_encoder"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(BASE_PATH, "../models/scaler"), "rb") as f:
        scaler = pickle.load(f)

    if st.button("Predict Delivery Time"):
        single_row_preprocessed = data_preprocessing(single_row_df, train=False, encoder=encoder, scaler=scaler)

        # Make a prediction using the pre-trained model
        prediction = model.predict(single_row_preprocessed)
        st.session_state.predicted_time = prediction[0]  # Save the prediction to session state

    # Display the prediction 
    if st.session_state.predicted_time is not None:
        st.markdown("### Estimated Delivery Time")
        st.markdown(
            f"""
            <div style="background-color: #FF6347; padding: 20px; border-radius: 10px; text-align: center;">
                <h1 style="color: white; font-size: 50px;">{st.session_state.predicted_time:.2f} minutes</h1>
                <p style="color: white;">Your estimated delivery time is shown above!</p>
            </div>
            """, unsafe_allow_html=True
        )

else:
    category = st.selectbox("Select a category:", list(menu.keys()))
    if category:
        selected_items = menu[category]
        selected_item = st.selectbox("Select an item:", selected_items)

        if selected_item:
            st.write(f"You selected: **{selected_item}** from the **{category}** menu.")
            st.info("Now, please select your restaurant location on the map.")

            # Set initial location to Tempe, AZ
            initial_lat, initial_lon = 33.4261, -111.9398
            # Create a Folium map centered at the initial location
            m = folium.Map(location=[initial_lat, initial_lon], zoom_start=12)
            # Add click event listener to the map
            m.add_child(folium.LatLngPopup())
            # Display the map using streamlit-folium
            st_data = st_folium(m, width=700, height=500)

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
                        st.session_state.delivery_location = None 
                    else:
                        st.session_state.delivery_selected = True  

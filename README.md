Estimating Delivery Time Prediction with Machine Learning and Real-Time Data
In this project, I focused on improving food delivery time predictions using machine learning and real-time data to enhance customer experience and operational efficiency. The goal was to predict delivery times more accurately, considering dynamic conditions such as traffic, weather, and delivery person performance.

Table of Contents
Overview
Real-Time Data Ingestion with Kafka
Data Cleaning and Feature Engineering
Key Insights
Modeling with LightGBM
Deployment with Streamlit
Conclusion
Overview
The project aims to create a more reliable and accurate food delivery time prediction system by utilizing machine learning models and real-time data. By factoring in various dynamic factors such as traffic density, weather conditions, and delivery person characteristics, the system estimates delivery time, improving the experience for customers and optimizing delivery operations for businesses.

Real-Time Data Ingestion with Kafka
To simulate the live data flow of food orders, I implemented Kafka to handle streaming data, replicating real-world conditions where new customer orders and delivery patterns continuously evolve.

This enables the system to ingest real-time data from various sources, ensuring that predictions are based on up-to-date information.

Data Cleaning and Feature Engineering
Handling raw data was critical to the success of the project. Key steps involved:

Handling Missing Values: Addressed missing values, zero entries, and duplicates.
Distance Calculation: Computed the distance between restaurants and delivery locations using the Haversine formula, providing an essential feature for delivery time prediction.
Time-Based Features: Created features such as order preparation time, pickup time, and differences between order times.
Delivery Person Characteristics: Features like the age and ratings of delivery personnel were generated, as these directly influenced delivery performance.
Key Insights
Several key insights were drawn from the data, including the impact of various factors on delivery times:

Traffic and Weather Impact: Delivery times were longest during foggy weather combined with heavy traffic, which significantly delayed delivery.
Delivery Person Age and Ratings: Delivery persons over 30 years of age, and those with higher ratings, consistently delivered faster than younger or lower-rated drivers.
City Type and Delivery Speed: Deliveries in semi-urban areas took considerably longer than in urban and metropolitan areas due to increased distances and infrastructure limitations.
Vehicle Condition: Poorly maintained vehicles caused longer delivery times, emphasizing the importance of proper vehicle upkeep.
Modeling with LightGBM
After evaluating several machine learning models, LightGBM was selected as the most effective model for predicting delivery time. It excelled due to its ability to handle both categorical and continuous features efficiently, and its robust performance in real-time data scenarios.

Accuracy: LightGBM handled the skewed distribution of the target variable (delivery time) and processed large datasets quickly.
Real-Time Adaptability: The model was highly suitable for real-time streaming data, providing accurate predictions even under dynamic conditions.
Deployment with Streamlit
To make the model user-friendly and accessible, I deployed the solution using Streamlit. This web application allows users to input real-time data such as traffic density, weather conditions, and delivery ratings, returning a predicted delivery time.

The app provides both customers and restaurants with a highly accurate estimation of when food will arrive, based on live conditions.

Conclusion
This project highlights the power of combining machine learning with real-time data to create a more efficient and customer-focused delivery prediction system. By accurately estimating delivery times, the project provides significant value to food delivery services, improving operational efficiency and customer satisfaction.

Tech Stack
Python
Kafka for real-time data streaming
Pandas, NumPy for data preprocessing and cleaning
LightGBM for machine learning model
Streamlit for deploying the web application
How to Run
Install required dependencies:
bash
Copy code
pip install -r requirements.txt
Start Kafka for data streaming.

Run the Streamlit app:
streamlit run Delivery_app_UI.py

bash
Copy code
streamlit run app.py
Input conditions such as traffic, weather, and delivery ratings in the web app, and the model will predict delivery time based on the input.
Future Improvements
Extend the model to include new features such as restaurant performance or customer behavior trends.
Improve real-time streaming capabilities to handle even larger volumes of data.
Integrate GPS tracking data for real-time location-based delivery estimation.

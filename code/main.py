import pandas as pd
import numpy as np
from data_preprocessing import preprocessed_data
from data_modelling import modelling
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    ##Load data
    logging.info("Loading data from CSV file")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "../data/uber-eats-deliveries.csv")
    print(data_path)
    df = pd.read_csv(data_path)
    print(df.shape)

    ## data preprocessing and saving the data
    ## Refer data_preprocessing.py script for the preprocesing done on raw data
    ## Pre processing data
    logging.info("Starting data preprocessing")
    preprocessed_df= preprocessed_data(df)
    preprocessed_df.to_csv(os.path.join(BASE_DIR,"../data/preprocessed_data.csv"), index = False)
    logging.info("Data preprocessing completed and saved to preprocessed_data.csv")

    ## Data modelling
    logging.info("Starting data modeling")
    model, encoder, scaler = modelling(preprocessed_df)
    logging.info("Data modeling completed")
    return model, encoder, scaler


if __name__ == '__main__':
    model, encoder, scaler = main()
    



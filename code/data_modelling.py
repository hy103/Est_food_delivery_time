import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
import pickle

### Packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,explained_variance_score, max_error
import warnings
import  lightgbm as lgb



def features_target_separation(data):
    ### Features and Predictors(Target)
    X = data.drop(['Time_taken(min)'], axis =1)
    y = data['Time_taken(min)']
    return X, y

def splitting_dataset(X, y, val_size, test_size):
    ### Splitting the data into Train validation and test
    X_train, X_test,y_train ,  y_test = train_test_split(X, y, test_size = val_size, random_state = 42)
    X_Val,  X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = test_size, random_state = 42)
    return X_train, X_Val, X_test, y_train, y_val, y_test



def convert_categorical_numerical_single(df, categorical_columns, fitted_encoder):
    # Transform using the pre-fitted encoder
    one_hot_encoded = fitted_encoder.transform(df[categorical_columns])
    one_hot_data = pd.DataFrame(one_hot_encoded, 
                                columns=fitted_encoder.get_feature_names_out(categorical_columns))
    
    df_rest_index = df.reset_index(drop=True, inplace=False)
    df_encoded = pd.concat([df_rest_index, one_hot_data], axis=1)
    df_encoded = df_encoded.drop(columns=categorical_columns, axis=1)
    return df_encoded

def feature_scaling_numeric_columns_single(df, numerical_columns, fitted_scaler):
    df[numerical_columns] = fitted_scaler.transform(df[numerical_columns])
    return df

def data_preprocessing(data, train=True, encoder = None, scaler = None):
    categorical_features = data.select_dtypes(include=['object']).columns
    if train == True:
        encoder = OneHotEncoder(sparse_output = False, handle_unknown='error')
        one_hot_encoded = encoder.fit_transform(data[categorical_features])
        one_hot_data = pd.DataFrame(one_hot_encoded, 
                                    columns = encoder.get_feature_names_out(categorical_features)) 
        df_rest_index = data.reset_index(drop = True, inplace = False)   
        df_encoded = pd.concat([df_rest_index, one_hot_data], axis =1)
        df_encoded = df_encoded.drop(columns = categorical_features, axis =1)

        numerical_features = df_encoded.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
        return df_encoded, encoder, scaler
    
    else:
        data_encoded = convert_categorical_numerical_single(data, categorical_features, encoder)
        numerical_features = data_encoded.select_dtypes(include=['float64', 'int64']).columns
        data_scaled = feature_scaling_numeric_columns_single(data_encoded, numerical_features, scaler)

        return data_scaled
    

### Applying various ML models
def model_training(X_train, X_test, y_train, y_test, ml_model):
    # Train the model
    ml_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = ml_model.predict(X_train)
    y_pred_test = ml_model.predict(X_test)

    # Evaluate the model
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared= False) 

    print(f"Model: {ml_model.__class__.__name__}")
    print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, R2 Score: {test_r2:.4f}")

    print('-' * 50)

    return ml_model, test_mse, test_r2, test_rmse


def training_for_best_model(X_train_pre, X_Val_pre, X_test_pre, y_train, y_val, y_test):
    models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, random_state=42),
        XGBRegressor(n_estimators=100, random_state=42),
        LGBMRegressor(n_estimators=100, random_state=42)
    ]

    # Train and evaluate each model
    trained_models = {}
    for model in models:
        trained_model, test_mse, r2,rmse = model_training(X_train_pre, X_Val_pre, y_train, y_val, model)
        trained_models[model.__class__.__name__] = {'model': trained_model, 'val_mse': test_mse, 'r2_score': r2, 'rmse': rmse}


    for model in models:
        trained_model, test_mse, r2,rmse = model_training(X_train_pre, X_test_pre, y_train, y_test, model)
        trained_models[model.__class__.__name__] = {'model': trained_model, 'test_mse': test_mse, 'r2_score': r2, 'rmse': rmse}

def best_model(X_train_pre, X_test_pre,y_train, y_test):
    hyper_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'feature_fraction': 0.9
    }


    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(X_train_pre, y_train,
            eval_set=[(X_test_pre, y_test)],
            eval_metric='l1')
    return gbm

def GetMetrics(y, predictions):
    return {'MSE' : mean_squared_error(y, predictions),
            'RMSE' : np.sqrt(mean_squared_error(y, predictions)),
            'MAE': mean_absolute_error(y, predictions)}


def visualizations(y_test, y_pred_test, ComparationTable):
    plt.scatter(y_test, y_pred_test)

    # Perfect predictions
    plt.plot(y_test, y_test, 'r')
    plt.ylabel('Real Value')
    plt.xlabel('Prediction')
    plt.title('Model Predictions vs. Perfect Predictions (Line)');

    errors = y_test.values.reshape(-1, 1) - y_pred_test
    sns.distplot(errors)
    plt.title('Residuals');

    fig, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 3))
    sns.histplot(data = ComparationTable, x = 'Difference %', ax = ax1)
    ax1.set_xlim(left = 0)
    ax1.set_title('Predictions Difference % to real value')

    sns.boxplot(data = ComparationTable, x = 'Difference %', ax = ax2)
    ax2.set_title('Predictions Difference % to real value');

    sns.histplot(data = ComparationTable, x = 'Real Value', label = 'Real Values', edgecolor = 'k')
    sns.histplot(data = ComparationTable, x = 'Model Prediction', label = 'Predictions', edgecolor = 'k')
    plt.legend();

    fig, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1,figsize = (15, 4))
    sns.countplot(data = ComparationTable, x = 'Model Prediction', ax = ax1, edgecolor = 'k', palette = 'Blues')
    ax1.set_title('Model predictions count')
    sns.countplot(data = ComparationTable, x = 'Real Value', ax = ax2, edgecolor = 'k', palette = 'Purples')
    ax2.set_title('Real values count')

    for ax in [ax1, ax2]:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    
    sns.countplot(data = ComparationTable, x = 'Difference', edgecolor = 'k', palette = 'Pastel1')
    plt.xticks(rotation = 90)
    plt.title('Difference count');

def metrics(best_ml_model, X_test_pre,y_test ):
        y_pred_test = best_ml_model.predict(X_test_pre)

        # Evaluate the model
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared= False) 

        print(f"Model: {best_ml_model.__class__.__name__}")
        print(f"Test MSE: {test_mse:.4f}, R2 Score: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        Summary = pd.DataFrame(GetMetrics(y = y_test, predictions = y_pred_test), index = ['Score'])

        Summary['Explained Variance'] = explained_variance_score(y_test, y_pred_test)
        Summary['Max Error'] = max_error(y_test, y_pred_test)


        ComparationTable = pd.DataFrame({
    'Real Value' : y_test.values,
    'Model Prediction' : [round(item) for item in y_pred_test],
    'Difference' : y_test.values - [round(item) for item in y_pred_test],
    'Difference %' : np.absolute((y_test.values - [round(item) for item in y_pred_test]) / y_test.values * 100)})
            
        #uncomment below line to plot the error and metrics
        #visualizations(y_test, y_pred_test, ComparationTable)


## Saving the model weights
def saving_models(best_ml_model, onehot_encoder, scaler, model_path):
    with open(os.path.join(model_path, "gbm.pkl"), "wb") as f:
        pickle.dump(best_ml_model, f)

    with open(os.path.join(model_path, "onehot_encoder"), "wb") as f:
        pickle.dump(onehot_encoder, f)
    with open(os.path.join(model_path, "scaler"), "wb") as f:
        pickle.dump(scaler, f)




def modelling(data):
    X, y = features_target_separation(data)
    X_train, X_Val, X_test, y_train, y_val, y_test = splitting_dataset(X, y, val_size = 0.2, test_size = 0.33)

    X_train_pre,encoder, scaler = data_preprocessing(X_train, train=True,encoder = None, scaler =None)
    X_Val_pre = data_preprocessing(X_Val, train=False, encoder = encoder, scaler =scaler)
    X_test_pre = data_preprocessing(X_test, train=False, encoder = encoder, scaler =scaler)
    ## Uncomment the below to train and test for the best model
    #training_for_best_model(X_train_pre, X_Val_pre, X_test_pre, y_train, y_val, y_test)
    best_ml_model = best_model(X_train_pre, X_test_pre,y_train, y_test)
    ## Uncomment the below line to see the train and test accuracies and plots
    #metrics(best_ml_model, X_test_pre,y_test )
    saving_models(best_ml_model, encoder, scaler, model_path = "../models")

    return best_ml_model, encoder, scaler

if __name__ == "__main__":
    ## Uncomment below lines if you want to test data modelling script alone
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #preprocessed_data = pd.read_csv(os.path.join(BASE_DIR,"../data/preprocessed_data.csv"), index = False)
    #best_ml_model, encoder, scaler = modelling(preprocessed_data)
    pass
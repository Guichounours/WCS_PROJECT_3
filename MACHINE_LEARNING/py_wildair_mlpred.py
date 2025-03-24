##### LIBRARIES IN USE #########

##### To get the Open-Meteo data #####
import openmeteo_requests
import requests_cache
from retry_requests import retry

##### Working the Data ######
import pandas as pd
import numpy as np

##### DataViz #####
import matplotlib.pyplot as plt
import seaborn as sns

##### Machine Learning #####
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV

#####################################

#### 1. GETTING THE ¨REVIOUSLY PREPARED HISTORIC DATASET MERGING METEO AND POLLUTANTS:

#### saving the CSV dataset in an usable pandas dataframe objet: 
df_metepollu_trainingset = pd.read_csv(r"C:\Users\sophi\FrMarques\LyonData WCS new\P3 wildAir\p3_WildAir\Open_Meteo_com\OpenMeteo_data\CSV\CSV_meteopollu_final\250219_meteopolluwind2124_buckets.csv")

#### Droping and reorganizing the columns: from a shape of (1461, 21) to (1461, 14) 
df_metepollu_trainingset = df_metepollu_trainingset[['date_id', 'temp_c', \
    'humidity_%', 'rain_mm', 'snowfall_cm', 'atmopressure_hpa', 'cloudcover_%', 'u10', 'v10', \
       'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']]

#### Setting all the columns labels in lower case/miniscules:
df_metepollu_trainingset.rename(columns=str.lower, inplace=True)

#### Setting the "date_id" column as 'datetime' and as 'index':
df_metepollu_trainingset["date_id"] = pd.to_datetime(df_metepollu_trainingset["date_id"])
df_metepollu_trainingset.set_index("date_id", inplace=True)

#### Saving the training dataset into a new CSV:
df_metepollu_trainingset.to_csv("meteo_pollu_trainingset.csv", index=True)

#########################################

#### 2. REQUESTING A NEW FORECAST TO OPEN-METEO 
## Example Source: https://open-meteo.com/en/docs#latitude=45.756&longitude=4.827&current=&minutely_15=&hourly=temperature_2m,relative_humidity_2m,rain,snowfall,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m&daily=&timezone=Europe%2FBerlin&forecast_days=14&models=

#### Open-Meteo API request to get a meteo forecast for a time window of 19fev-04mars 2025:
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 45.756,
	"longitude": 4.827,
	"hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
	"timezone": "Europe/Berlin",
	"forecast_days": 14
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(7).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe)

#### Converting the timezone from GMT to CET (even setting the request for "Europe/Berlin", the data 
#          arrives in a timezone of GMT+0)
hourly_dataframe["date"] = pd.to_datetime(hourly_dataframe["date"]).dt.tz_convert("Europe/Paris")
hourly_dataframe.head(2)

#### Adding a new "date_only" column to allow a group by 'day' with 'mean' values 
#           (original date column is droped and "date_only" is set as datetime dtype and the new 'index'):
hourly_dataframe["date_only"] = hourly_dataframe["date"].dt.date
daily_dataframe = hourly_dataframe.groupby("date_only").mean().reset_index()
daily_dataframe["date_only"] = pd.to_datetime(daily_dataframe["date_only"])
daily_dataframe = daily_dataframe.drop(columns="date")
daily_dataframe.set_index("date_only", inplace=True)

#### Renaming all columns to harmonize the datasets

daily_dataframe = daily_dataframe.rename(columns={
    "date_only": "date_id",
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_%",
    "rain": "rain_mm",
    "snowfall": "snowfall_cm",
    "surface_pressure": "atmopressure_hpa",
    "cloud_cover": "cloudcover_%",
    "wind_speed_10m": "windspeed_kmh",
    "wind_direction_10m": "winddirection_360"
    })

#### Converting wind data into numeric values that Machine Learning can work on and droping unecessary columns:

daily_dataframe["windspeed_ms"] = daily_dataframe["windspeed_kmh"] * 0.277778      # turning wind speed from km/h to m/s;
wind_dir_rad = np.radians(daily_dataframe["winddirection_360"])                    # turning direction to radians;
daily_dataframe["u10"] = -daily_dataframe["windspeed_ms"] * np.sin(wind_dir_rad)   # calculating "u" and "v" from the horizontal (cos/ N->S) and
daily_dataframe["v10"] = -daily_dataframe["windspeed_ms"] * np.cos(wind_dir_rad)   #.... vertical (sin/ E->W) unities for wind;
columns2drop = ["windspeed_ms", "windspeed_kmh", "winddirection_360"]              # droping the now unecessary columns;
daily_dataframe = daily_dataframe.drop(columns=columns2drop)

#### Saving the resulting dataframe into a CSV file:

daily_dataframe.to_csv("df_meteoforecast_19fev_04mar25.csv", index=True)

###############################################################

#### 3. MACHINE LEARNING PREPS

#### 3.1. Calling in the historic dataset to train the model:

df_training_2124 = pd.read_csv(r"C:\Users\sophi\FrMarques\LyonData WCS new\P3 wildAir\p3_WildAir\Open_Meteo_com\OpenMeteo_data\CSV\CSV_meteopollu_final\meteo_pollu_trainingset.csv")

#### 3.2. Setting the "date_id" as index to make it invisible for the ML models:

df_training_2124["date_id"] = pd.to_datetime(df_training_2124["date_id"])
df_training_2124.set_index("date_id", inplace=True)

##################################################################

#### 4. MACHINE LEARNING: Random Forest Regressor optimized  train

df_train = df_training_2124.copy()                              # Creating a new copy from the historic data to
features = ['temp_c', 'humidity_%', 'rain_mm', 'snowfall_cm',   #... avoid errors in the original "df";
            'atmopressure_hpa', 'cloudcover_%', 'u10', 'v10']

pollutants = ["no2", "o3", "pm10", "pm2.5", "so2"]

X = df_train[features]      # setting the "X" with the meteo features with correlations with the target;
y_dict = {pollutant: df_train[pollutant] for pollutant in pollutants}   # setting the target/ pollutants as "key: value" or "label: value";

X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {} # empty dict to save the train test dict groups;

for pollutant in pollutants:        # iteration to create the "train" and "test" teams (80%/20%) for each pollutant:
    X_train[pollutant], X_test[pollutant], y_train_dict[pollutant], y_test_dict[pollutant] = train_test_split(
        X, y_dict[pollutant], test_size=0.2, random_state=42    #... ensuring the samples are reused and not shuffled;
    )

best_params = {             # setting the best parameters tuned before with 
    "n_estimators": 100,    #... a 100 trees/ estimators/ decision makers,
    "max_depth": 15,        #... that can go into a depth of 15,
    "min_samples_split": 2, #... with a minimum of 2 samples by node to split into a new branch,
    "min_samples_leaf": 5,  #... knowing that the final leaf/external node must have 5 samples
    "random_state": 42      #... and the samples are resuable and not shuffled.
}

rfr_models = {}             # empty dict to save the model for each pollutant;
metrics = {}                # empty dict to save the metrics for each pollutant;

for pollutant in pollutants:    # loop to iterate on each pollutant and execute the RFR training, test and evaluation;
    print(f" Training the RFR for {pollutant}...")

    rfr = RandomForestRegressor(**best_params)
    rfr.fit(X_train[pollutant], y_train_dict[pollutant])

    rfr_models[pollutant] = rfr  

    y_pred = rfr.predict(X_test[pollutant])  

    mae = mean_absolute_error(y_test_dict[pollutant], y_pred)
    mse = mean_squared_error(y_test_dict[pollutant], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_dict[pollutant], y_pred)

    metrics[pollutant] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

print("\n Evaluating the final model:")     # Showing the evaluation results:

for pollutant, values in metrics.items():
    print(f"\n {pollutant}:")
    for metric, value in values.items():
        print(f"   -{metric}: {value:.2f}")

#### 4.1. Defining a function to predict pollutants from a meteo forecast dataset previously prepared: 

def predict_pollution(forecast_meteo, trained_models):
    """
    Function to forecast pollution levels from meteo forecasts.

    Parameters:
    - forecast_meteo (DataFrame): DataFrame with meteo forecasts.
    - trained_models (dict): Dictionary with the trained models {pollutant: correspondent model}.
    
    # Example of use:
    forecast_meteo = pd.read_csv("forecast_meteo_14days.csv")
    df_results = predict_pollution(forecast_meteo, rfr_models)
    print(df_results)

    Returns:
    - df_predictions (DataFrame): DataFrame with forecasts for each pollutant.
    """
    
    df_predictions = forecast_meteo[["date_id"]].copy() # creating a dataframe to make the predictions by date;

    forecast_meteo = forecast_meteo.drop(columns=["date_id"], errors="ignore") # removes the "date_id" if its present in the forecast 'else' ignores the absence;

    for pollutant, model in trained_models.items(): # for each pollutant uses the trained model to predict;
        print(f" Calculating a forecast for {pollutant}...")
        df_predictions[f"{pollutant}_Predicted"] = model.predict(forecast_meteo)

    return df_predictions       # returns a dataframe with the new pollutants predictions columns for each given date.

###########################################################

#### 5. PREDICTING POLLUTANTS LEVEL TO SET THE QAI (AIR QUALITY INDEX)/ IQA IN FRENCH

#### 5.1. creating a pandas dataframe with the Open-Meteo forecast data previously prepared:
df_predict_pm25 = pd.read_csv(r"C:\Users\sophi\FrMarques\LyonData WCS new\P3 wildAir\p3_WildAir\Prediction_final_ML\CSV to predict\df_meteoforecast_19fev_04mar25.csv")

#### 5.2. resetting the index to allow the "date" to be used by the model:
df_predict_pm25.reset_index(inplace=True)

#### 5.3 applying the prediction function and showing the results:
df_forecast_19fev_04mars25 = predict_pollution(df_predict_pm25, rfr_models)
df_forecast_19fev_04mars25



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime
import pytz

import pandas as pd
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(pickup_datetime,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            passenger_count
            ):
    # change variable types
    pickup_datetime = str(pickup_datetime)
    pickup_longitude = float(pickup_longitude)
    pickup_latitude = float(pickup_latitude)
    dropoff_longitude = float(dropoff_longitude)
    dropoff_latitude = float(dropoff_latitude)
    passenger_count = int(passenger_count)
    # fix the datetime :
    # create a datetime object from the user provided datetime
    # and localize the user datetime with NYC timezone
    # and localize the datetime to UTC
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    eastern = pytz.timezone("US/Eastern")
    pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    pickup_datetime = pickup_datetime.astimezone(pytz.utc)
    pickup_datetime = pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
    # create X_pred as a Dataframe
    X_pred = pd.DataFrame({
        "key" : ["2000-01-01 00:00:00.0000000"],
        "pickup_datetime" : [pickup_datetime],
        "pickup_longitude" : [pickup_longitude],
        "pickup_latitude" : [pickup_latitude],
        "dropoff_longitude" : [dropoff_longitude],
        "dropoff_latitude" : [dropoff_latitude],
        "passenger_count" : [passenger_count]
    })
    # load the pre-trained model
    try:
        model = joblib.load("model.joblib")
    except FileNotFoundError:
        return {"fare" : -1}
    # compute prediction
    y_pred = model.predict(X_pred)
    # return predicted
    return {"fare" : round(y_pred[0], 2)}

if __name__ == "__main__":
    print(predict(
        "2013-07-06 17:18:00",
        "-73.950655",
        "40.783282",
        "-73.984365",
        "40.769802",
        "1"
    ))

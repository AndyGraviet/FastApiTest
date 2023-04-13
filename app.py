import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List
from sklearn import preprocessing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the API!"}

class Data(BaseModel):
    data: List[int]

# Define endpoint for making predictions
@app.post('/predict')
def predict(data: Data):
    # Load model from .pkl file
    with open('./WrappingModel.pkl','rb') as file:
        model = pickle.load(file)
        # Convert input data to DataFrame
        data_array = np.array(data.data).reshape(1, -1)
        # Make prediction
        prediction = model.predict(data_array)
        # Return Prediction as JSON response
        return {'prediction': prediction[0]}
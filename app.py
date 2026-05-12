import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

# MLflow tracking server
mlflow.set_tracking_uri("http://host.docker.internal:5000")

# Load model from MLflow Registry
model = mlflow.sklearn.load_model("models:/iris_model/1")

# FastAPI app
app = FastAPI()

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction API
@app.post("/predict")
def predict(data: IrisInput):

    sample = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(sample)

    return {
        "prediction": int(prediction[0])
    }
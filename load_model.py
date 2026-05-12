import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.sklearn.load_model("models:/iris_model/1")

print("Model loaded ✔️")
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print("Prediction:", prediction)
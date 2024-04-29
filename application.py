from flask import Flask, request, render_template, url_for
from src.pipeline.predict_pipeline import CustomData, Predict_Pipeline
import requests
import json
from src.api.api_model import predict

application = Flask(__name__)


# Creating route for index page
@application.route("/")
def index():
    return render_template("index.html")




@application.route("/prediction", methods=["POST", "GET"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data_pipeline = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("writing_score")),
            writing_score=float(request.form.get("reading_score")),
        )
        features = data_pipeline.get_data_as_data_frame()
        pred_pipeline = Predict_Pipeline()
        data = pred_pipeline.predict(features=features)
        return render_template("home.html", results=data)


if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=True)

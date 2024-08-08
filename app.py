import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask application
application = Flask(__name__)
app = application  # Assign the Flask instance to 'app'


# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Route for making predictions
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            # Create a CustomData instance from form data
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethinicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get(
                    "parental_level_of_education"
                ),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("reading_score")),
                writing_score=float(request.form.get("writing_score")),
            )

            # Convert the data to a DataFrame
            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            # Initialize the prediction pipeline and make predictions
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Render the result in the home template
            return render_template("home.html", results=results[0])

        except ValueError as e:
            print(f"Error converting input data to float: {e}")
            return render_template(
                "home.html",
                error="Invalid input. Please check your entries and try again.",
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return render_template(
                "home.html", error="An error occurred. Please try again later."
            )


# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

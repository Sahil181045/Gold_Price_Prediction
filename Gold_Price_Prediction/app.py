
from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/display_gold", methods=['POST'])
def show_gold():
    try:
        if request.method == "POST":
            SPX = float(request.form["SPX"])
            USO = float(request.form["USO"])
            SLV = float(request.form["SLV"])
            EUR_USD = float(request.form["EUR_USD"])

            input_data = (SPX, USO, SLV, EUR_USD)
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            with open('gold_predictor.pkl', 'rb') as f:
                regressor = joblib.load(f)

            prediction = regressor.predict(input_data_reshaped)[0]
            prediction = "{:.2f}".format(prediction)

            return render_template("display_gold.html", prediction=prediction)
    except Exception as e:
        return "<h1> Something went wrong </h1>"
    

@app.route("/display_silver", methods=['POST'])
def show_silver():
    try:
        if request.method == "POST":
            SPX = float(request.form["SPX"])
            GLD = float(request.form["GLD"])
            USO = float(request.form["USO"])
            EUR_USD = float(request.form["EUR_USD"])

            input_data = (SPX, GLD, USO, EUR_USD)
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            with open('silver_predictor.pkl', 'rb') as f:
                regressor = joblib.load(f)

            prediction = regressor.predict(input_data_reshaped)[0]
            prediction = "{:.2f}".format(prediction)

            return render_template("display_silver.html", prediction=prediction)
    except Exception as e:
        return "<h1> Something went wrong </h1>"
    


if __name__ == "__main__":
    app.run(debug=True)

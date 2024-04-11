from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("fitted_random_forest.pkl")
preprocessor = joblib.load("fitted_pipeline.pkl")
cols = ['ACCLOC', 'TRAFFCTL', 'LIGHT', 'RDSFCOND', 'IMPACTYPE',
       'INVTYPE', 'INVAGE', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND',
       'AG_DRIV', 'HOOD_158']

# route the app
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    print("Received form submission")
    
    # Retrieve form data
    ACCLOC = np.array([request.form['ACCLOC']])
    TRAFFCTL = np.array([request.form['TRAFFCTL']])
    LIGHT = np.array([request.form['LIGHT']])
    RDSFCOND = np.array([request.form['RDSFCOND']])
    IMPACTYPE = np.array([request.form['IMPACTYPE']])
    INVTYPE = np.array([request.form['INVTYPE']])
    INVAGE = np.array([request.form['INVAGE']])
    VEHTYPE = np.array([request.form['VEHTYPE']])
    MANOEUVER = np.array([request.form['MANOEUVER']])
    DRIVACT = np.array([request.form['DRIVACT']])
    DRIVCOND = np.array([request.form['DRIVCOND']])
    AG_DRIV = np.array([request.form['AG_DRIV']])
    HOOD_158 = np.array([request.form['HOOD_158']])
    
    # Concatenate form data
    final = np.concatenate([ACCLOC, TRAFFCTL, LIGHT, RDSFCOND, IMPACTYPE, INVTYPE,
                            INVAGE, VEHTYPE, MANOEUVER, DRIVACT, DRIVCOND, AG_DRIV,
                            HOOD_158])
    
    final = np.array(final)
    data = pd.DataFrame([final], columns=cols)
    data_trans = preprocessor.transform(data)
    data_reshaped = data_trans.reshape(1, -1)
    prediction = model.predict(data_reshaped)
    return render_template("result.html", prediction=prediction[0])
    

if __name__ == "__main__":
    app.run(debug=True, port=5000)

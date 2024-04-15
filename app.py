from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

#---------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

#---------------------------------------------------------

@app.route('/cat_fields.json')
def cat_fields():
    with open('cat_fields.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)

#---------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    params_dict = {}
    # Get all entries from form into dictionary
    for field_name in request.form:
        params_dict[field_name.upper()] = request.form[field_name]
    # Seperate the model from the features
    model_type = params_dict.pop('MODEL')
    # Convert the entries for the remaining features to a df
    params_df = pd.DataFrame([params_dict])


    print(params_dict)
    print(model_type)
    print(params_df)

    pipeline = joblib.load(open('fitted_pipeline.pkl','rb'))
    print(type(pipeline))

    trans_params = pipeline.transform(params_df)

    print(trans_params.shape)

    if model_type == 'nn':
        model = joblib.load(open('fitted_neural_network.pkl','rb'))
        name = "Neural Network"
        testing_accuracy = "85.6%"
    elif model_type == 'rf':
        model = joblib.load(open('fitted_random_forest.pkl','rb'))
        name = "Random Forest"
        testing_accuracy = "87.8%"
    elif model_type == 'svm':
        model = joblib.load(open('fitted_svm.pkl','rb'))
        name = "Support Vector Machine"
        testing_accuracy = "84.0%"
    elif model_type == 'ensemble':
        model = joblib.load(open('fitted_ensemble_learning.pkl','rb'))
        name = "Support Vector Machine"
        testing_accuracy = "87.7%"
   
    prediction = model.predict(trans_params)[0]
    
    print(prediction)
    print(type(prediction[0]))

    #prediction_proba = model.predict_proba(trans_params)
    #print(prediction_proba)

    print('-'*50)

    print('Form submitted successfully')

    return render_template('results.html', name=name, testing_accuracy=testing_accuracy, prediction=prediction)
#---------------------------------------------------------



#---------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)

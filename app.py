from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
cors = CORS(app,resource={
    r"/*":{
        "origins":"*"
    }
})

@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route("/predict_BCancer" , methods=['POST'])
def predict_BCancer():
    data = request.get_json()
    # print(data)
    # print(data['mean radius'])
    df_os = pd.DataFrame(data, index=[0])
    # print(df_os['mean radius'])
    prediction = np.array2string(model.predict(df_os))
    print(prediction[1])
    return jsonify(prediction[1])
#diabeties
@app.route("/predict_diabeties" , methods=['POST'])
def predict_diabeties():
    Pregnancies = float(request.args.get("Pregnancies"))
    Glucose = float(request.args.get("Glucose"))
    BloodPressure = float(request.args.get("BloodPressure"))
    SkinThickness = float(request.args.get("SkinThickness"))
    Insulin = float(request.args.get("Insulin"))
    BMI = float(request.args.get("BMI"))
    DiabetesPedigreeFunction = float(request.args.get("DiabetesPedigreeFunction"))
    Age = float(request.args.get("Age"))
    
    print(type(Pregnancies))

    tupled_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    print(tupled_data)
    print(type(tupled_data))

    array_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    print(array_data)
    print(type(array_data))
    input_data_as_numpy_array = np.asarray(array_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction_a = model_dib.predict(input_data_reshaped)
    print(prediction_a)

    if (prediction_a[0] == 0):
        return 'The person is not diabetic'
    else:
        return'The person is diabetic'
    # return "1"

@app.route("/predict_heart_disease" , methods=['POST'])
def predict_heart_disease():
    age = float(request.args.get("age"))
    sex = float(request.args.get("sex"))
    cp = float(request.args.get("cp"))
    trestbps = float(request.args.get("trestbps"))
    chol = float(request.args.get("chol"))
    fbs = float(request.args.get("fbs"))
    restecg = float(request.args.get("restecg"))
    thalach = float(request.args.get("thalach"))
    exang = float(request.args.get("exang"))
    oldpeak = float(request.args.get("oldpeak"))
    slope = float(request.args.get("slope"))
    ca = float(request.args.get("ca"))
    thal = float(request.args.get("thal"))

    print(thal)

    array_data_HD = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    print(array_data_HD)
    print(type(array_data_HD))
    input_data_as_numpy_array_HD = np.asarray(array_data_HD)

    #reshape the array as we are predicting for one instance
    input_data_reshaped_HD = input_data_as_numpy_array_HD.reshape(1,-1)
    prediction_a_HD = model_heart_disease.predict(input_data_reshaped_HD)
    print(prediction_a_HD)

    if (prediction_a_HD[0] == 0):
        return "0"
    else:
        return"1"

@app.route("/predict_brain_disease" , methods=['POST'])
def predict_brain_disease():
    brain_disease_data = request.get_json()
    print(brain_disease_data)
    br_df_os = pd.DataFrame(brain_disease_data, index=[0])
    # print(df_os['mean radius'])
    br_prediction = np.array2string(model_brain.predict(br_df_os))
    print(br_prediction[1])
    #"[0]"
    return jsonify(br_prediction[1])
    # return "1"

if __name__ == '__main__':
    modelfile = 'breast_cancer_detector.pickle'
    model = p.load(open(modelfile, 'rb'))
    dib_modelfile = 'diabeties_detector.pickle'
    model_dib = p.load(open(dib_modelfile, 'rb'))
    heart_disease_modelfile = 'heart_disease_detector.pickle'
    model_heart_disease = p.load(open(heart_disease_modelfile, 'rb'))
    model_br = 'brain_disease_detector.pickle'
    model_brain = p.load(open(model_br, 'rb'))
    print("model loded")
    app.run(debug=True, host='0.0.0.0')
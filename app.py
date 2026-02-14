from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline + column order
model = joblib.load("heart_logistic_model.pkl")
columns = joblib.load("columns.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():

    #Convert form inputs
    data = {
        "age": float(request.form['age']),
        "sex": 0 if request.form['gender'] == "male" else 1,
        "cp": float(request.form['cp']),
        "trestbps": float(request.form['trestbps']),
        "chol": float(request.form['chol']),
        "fbs": float(request.form['fbs']),
        "restecg": float(request.form['restecg']),
        "thalach": float(request.form['thalach']),
        "exang": 1 if request.form['exang'] == "Yes" else 0,
        "oldpeak": float(request.form['oldpeak']),
        "slope": float(request.form['slope']),
        "ca": float(request.form['ca']),
        "thal": float(request.form['thal'])
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Ensure correct column order
    df = df.reindex(columns=columns, fill_value=0)

    #Prediction calculation
    target_result = model.predict(df)[0]
    prediction=""
    
    #output correct message
    #use of predict_proba() method which predicts the probability if either classification
    if target_result==1:
        prediction = "The patient is very likely to have heart disease."
        probability = model.predict_proba(df)[0][1]
    elif target_result==0:
        prediction="The patient is unlikely to have heart disease."
        probability = model.predict_proba(df)[0][0]

    

    return render_template(
        'results.html',
        prediction=prediction,
        probability=round(probability * 100, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)


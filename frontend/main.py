from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        form_data = {
            'age': int(request.form['age']),
            'sex': int(request.form['sex']),
            'chestpaintype': int(request.form['chestpaintype']),
            'trestbps': int(request.form['trestbps']),
            'cholesterol': int(request.form['cholesterol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'exang': int(request.form['exang']),
            'stdepression': float(request.form['stdepression']),
            'stslope': int(request.form['stslope']),
            'num_major_vessels': int(request.form['num_major_vessels']),
            'thal': int(request.form['thal'])
        }

        input_data = pd.DataFrame([form_data])

        input_data = input_data[['age', 'sex', 'chestpaintype', 'trestbps', 'cholesterol', 
                                 'fbs', 'restecg', 'thalach', 'exang', 'stdepression', 
                                 'stslope', 'num_major_vessels', 'thal']]

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        result = 'have heart disease' if prediction == 1 else 'do not have heart disease'
        
        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

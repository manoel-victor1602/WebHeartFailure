from flask import Flask, render_template, request
import joblib
import numpy as np

encoder_path = 'encoders/'

app = Flask(__name__)

@app.route('/')
def student():
   return render_template('input.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		result = request.form
		prediction, = classify(result)
		prediction = resolve(prediction)

		return render_template("result.html", result = result, prediction = prediction)

def classify(result):
	hf_encoder = joblib.load(encoder_path + 'hf_encoder.pkl')
	family_encoder = joblib.load(encoder_path + 'family_encoder.pkl')
	sex_encoder = joblib.load(encoder_path + 'sex_encoder.pkl')
	smoker_encoder = joblib.load(encoder_path + 'smoker_encoder.pkl')

	clf = joblib.load('rf.pkl')

	features = [result['AVGHEARTBEATSPERMIN'],
		       result['PALPITATIONSPERDAY'],
		       result['CHOLESTEROL'],
		       result['BMI'],
		       result['AGE'],
		       *sex_encoder.transform([result['SEX']]),
		       *family_encoder.transform([result['FAMILYHISTORY']]),
		       *smoker_encoder.transform([result['SMOKERLAST5YRS']]),
		       result['EXERCISEMINPERWEEK']
	]

	prediction = clf.predict(np.reshape(list(map(int, features)), (1, -1)))
	return hf_encoder.inverse_transform(prediction)

def resolve(prediction):
	if (prediction == 'N'):
		return "Não possui Falha Cardíaca"
	else:
		return "Possui Falha Cardíaca"

if __name__ == '__main__':
   app.run(debug = True)
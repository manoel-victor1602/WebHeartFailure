from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np

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
	hf_encoder = joblib.load('hf_encoder.pkl')
	family_encoder = joblib.load('family_encoder.pkl')
	sex_encoder = joblib.load('sex_encoder.pkl')
	smoker_encoder = joblib.load('smoker_encoder.pkl')

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
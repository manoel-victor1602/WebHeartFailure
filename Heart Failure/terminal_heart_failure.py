from sklearn.externals import joblib
import numpy as np

def resolve(prediction):
	if (prediction == 'N'):
		return "Não possui Falha Cardíaca"
	else:
		return "Possui Falha Cardíaca"

hf_encoder = joblib.load('hf_encoder.pkl')
family_encoder = joblib.load('family_encoder.pkl')
sex_encoder = joblib.load('sex_encoder.pkl')
smoker_encoder = joblib.load('smoker_encoder.pkl')

clf = joblib.load('rf.pkl')

avg_heartbeat = int(input('Batimento cardíaco médio por minuto: '))
palpitation = int(input('Palpitações por dia: '))
choleterol = int(input('Colesterol: '))
bmi = int(input('BMI: '))
age = int(input('Idade: '))
gender = sex_encoder.transform([input("Gênero: ")])[0]
family = family_encoder.transform([input('Historico familiar: ')])[0]
smoker = smoker_encoder.transform([input('Fumante: ')])[0]
exercise_min = int(input("Minutos de exercicios por semana: "))

features = [avg_heartbeat,
			palpitation,
			choleterol,
			bmi,
			age,
			gender,
			family,
			smoker,
			exercise_min
		]

prediction = clf.predict(np.reshape(list(map(int, features)), (1, -1)))[0]

print(resolve(prediction))
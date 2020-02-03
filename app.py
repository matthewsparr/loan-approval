import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.static_folder = 'static' 
model = pickle.load(open('model.pkl', 'rb'))
rate_model = pickle.load(open('rate_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
rate_scaler = pickle.load(open('rate_scaler.pkl', 'rb'))
states = pickle.load(open('states_list.pkl', 'rb'))
purposes = pickle.load(open('purposes_list.pkl', 'rb'))
home_owners = pickle.load(open('home_owner_list.pkl', 'rb'))
terms = [24, 36, 48, 60]
rate_interval_weight = 0.85

title = "Please enter the following information to check your loan eligibility:"
@app.route('/')
def home():
	return render_template('index.html', title=title, show_title=True, home_owners=home_owners, purposes=purposes, states=states, show_cards=False, show_forms=True)

@app.route('/predict',methods=['POST'])
def predict():
	first_loan = second_loan = True
	loan_amnt = float(request.form['loan_amnt'])
	annual_inc = float(request.form['annual_inc'])
	fico = int(request.form['fico'])
	emp_length = float(request.form['emp_length'])
	home_owner = str(request.form['home_ownership'])
	state = str(request.form['state'])
	purpose = str(request.form['loan_purpose'])
	joint_app = str(request.form['joint_application'])

	state_data = np.array([1 if state==i else 0 for i in states])
	purpose_data = np.array([1 if purpose==i else 0 for i in purposes])
	home_owner_data = np.array([1 if home_owner==i else 0 for i in home_owners])
	cosigner = np.array([joint_app])

	rate_pred = rate_model.predict(rate_scaler.transform(np.array([loan_amnt,fico]).reshape(1,-1))).round(2)
	rates = np.arange(start=rate_pred*(rate_interval_weight+1), stop=rate_pred*rate_interval_weight, step=-0.005)

	results = [] 
	term_available = []
	for term in terms:
		found_loan=False
		for rate in rates: 
			loan_data = np.array([loan_amnt, term, rate, annual_inc, fico, emp_length])
			application_data = np.hstack((loan_data, home_owner_data, purpose_data,  cosigner, state_data)).astype('float64')
			application_data = scaler.transform(application_data.reshape(1,-1))
			prediction = model.predict(application_data)[0]
			if not prediction:
				results.append((term,rate))
				found_loan = True
				term_available.append(True)
				break
		if not found_loan:
			term_available.append(False)

	if results == []: 
		results_text = "Sorry, but we cannot offer you a loan at this time."
		loan_amnt = apr1 = term1 = apr2 = term2 = monthly1 = monthly2 = 0
		show_cards = False
	else:
		results_text = "We can offer you the following loans:"
		show_cards = True
		loan_options = []
		for i in results:
			rate = i[1]
			months = i[0]
			apr = str((rate*100).round(2)) + "% APR"
			term = str(months) + " months"
			payment = "$" + str(int(loan_amnt * ((rate/12 * ((1+rate/12) ** months)) / ((1+rate/12) ** months - 1)))) + "/mo"
			loan_options.append((apr,term,payment))

	return render_template('index.html', title=title, show_cards=show_cards, show_title=False, show_forms=False, 
							results_text=results_text, home_owners=home_owners, purposes=purposes, states=states, 
							loan_amnt=int(loan_amnt), loan_options=loan_options)

if __name__ == "__main__":
	app.run(debug=True)
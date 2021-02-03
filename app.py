# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:24:41 2021

@author: Sindhu D
"""

from flask import Flask, render_template, request
import pickle

filename = 'random.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
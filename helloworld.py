import numpy as np 
from flask import Flask, render_template, jsonify, request
import pickle

app = Flask(__name__)
model = pickle.load(open('Jupyter Notebook\music-recommender.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    web_prediction = model.predict(final_features)

    output = web_prediction[0]

    return render_template('index.html', prediction_text='A genre that you will like is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False)

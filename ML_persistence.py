#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pickle

music_df = pd.read_csv('music.csv')
X = music_df.drop(columns=['genre'])
y = music_df['genre']

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.33)

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict(X_test)
predict_score = accuracy_score(y_test, predictions)

pickle.dump(model, open('music-recommender.pkl', 'wb'))

joblib.dump(model, 'music-recommender.joblib')

trained_model = joblib.load('music-recommender.joblib')
trained_predictions = trained_model.predict([[29, 0]])
print(trained_predictions)

trained_model_pkl = pickle.load(open('music-recommender.pkl', 'rb'))
trained_predictions = trained_model.predict([[29, 0]])
print(trained_predictions)


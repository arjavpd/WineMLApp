import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np




st.write("""
# Wine Class Prediction Model
""")

st.write('Using the wine [dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine) from sckit learn toy datasets, \nthis web app predicts the class of wine based on different features\ndetermined from the sidebar. Play around with the sliders to see different predictions. :) [Github](https://github.com/arjavpd/WineMLApp)')


st.sidebar.header('User Input Parameters')



def user_input_features():
    alcohol = st.sidebar.slider('Alcohol', 10.2, 14.83, 13.0)
    malic_acid = st.sidebar.slider('Malic acid', 0.01,5.8, 2.1)
    ash = st.sidebar.slider('Ash', 1.1, 3.32, 2.1)
    alcalinity_of_ash = st.sidebar.slider('Alcalinity of Ash', 14.3, 30.0, 19.3)
    magnesium = st.sidebar.slider('Magnesium', 60.4, 162.0, 90.4)
    total_phenols = st.sidebar.slider('Total Phenols', 1.01, 3.88, 2.29)
    flavanoids = st.sidebar.slider('Flavanoids', 0.1, 5.08, 2.29)
    nonflavanoid_phenols = st.sidebar.slider('Nonflavanoid Phenols', 0.1, 0.66, 0.36)
    proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.0, 3.58, 1.59)
    color_intensity = st.sidebar.slider('Color Intensity', 0.50, 13.1, 5.50)
    hue = st.sidebar.slider('Hue', 0.1, 1.71, 0.957)
    diluted_wines = st.sidebar.slider('od280/od315 of Diluted Wines', 0.2, 4.2, 1.27)
    proline = st.sidebar.slider('Proline', 0.1, 278.2, 1679.1)
    data = {'alcohol': alcohol,
            'malic_acid': malic_acid,
            'ash': ash,
            'alcalinity_of_ash': alcalinity_of_ash,
            'magnesium': magnesium,
            'total_phenols': total_phenols,
            'flavanoids': flavanoids,
            'nonflavanoid_phenols': nonflavanoid_phenols,
            'proanthocyanins': proanthocyanins,
            'color_intensity': color_intensity,
            'hue': hue,
            'od280/od315_of_diluted_wines': diluted_wines,
            'proline': proline}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)



wine = datasets.load_wine()
X = wine.data
Y = wine.target


clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(wine.target_names)

st.subheader('Prediction')
st.write(wine.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

 # Plot 
model = RandomForestRegressor()
model.fit(X, Y)
importance = model.feature_importances_
fig = plt.figure()
plt.bar([x for x in range(len(importance))], importance)
plt.title('Feature Importance using Random Forest Regression')
plt.xlabel('Feature')
plt.ylabel('Importance')


st.subheader('Feature Importance using Random Forest Regression')
st.pyplot(fig)
st.write('This graph shows that feature 6 (total phenols) is the most important feature\nwhen it comes to determining the class of wine.')



 

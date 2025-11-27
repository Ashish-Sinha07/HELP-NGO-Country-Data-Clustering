import streamlit as st
import numpy as np
import pandas as pd
import joblib



# Lets load the joblib instances over there
with open('pipeline.joblib','rb') as file:
    preprocess = joblib.load(file)

with open('model.joblib','rb') as file:
    model = joblib.load(file)

# lets take inputs from users
st.title('HELP NGO Organization')
st.subheader('This application will help to identify the development category of the country using socio economic factor. This application make predictions about the countries.  ')


# lets take the input
gpp = st.number_input('Enter the GDP of a country (GPP per pupulation)')
income = st.number_input('Enter income per population')
imports = st.number_input('Imports of goods and services per capita. Given as %age of the GDP per capita')
exports = st.number_input('Exports of goods and services per capita. Given as %age of the GDP per capita')
inflation = st.number_input('Inflation: Imports of goods and services per capita. Given as %age of the GDP per capita')

lf_expcy = st.number_input('Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain the same')
fert = st.number_input('Fertility: The number of children that would be born to each woman if the current age-fertility rates remain the same.')
health = st.number_input('Health: Total health spending per capita. Given as %age of GDP per capita')
child_mort = st.number_input('Child Mortality: Death of children under 5 years of age per 1000 live births')

input_list = [child_mort,exports,health,imports,income,inflation,lf_expcy,fert,gpp]

final_input_list = preprocess.transform([input_list])

if st.button('Predict'):
    prediction = model.predict(final_input_list)[0]
    if prediction == 0:
        st.success('Developing')
    elif prediction == 1:
        st.success('Developed')
    else:
        st.error('Underdeveloped')


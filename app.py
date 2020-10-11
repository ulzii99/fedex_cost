import streamlit as st
import pandas as pd
import numpy as np
import sklearn.model_selection as sk_ms
import statsmodels.formula.api as sm
import sklearn.metrics as sk_m


st.title("Let's predict the occupancy of the room. Lets go")
temp = st.sidebar.slider("Temperature", 19.0, 23.2)
hum = st.sidebar.slider("Humidity", 16.5, 39.2)

df_eco = pd.read_csv('C:/Users/ulzii/OneDrive/2. CBS semester 3/2. Business Analytics II/Class material/Module 2 - Powerful Predictions/Cases/Trump and EcoLights LLC/EcoLights.csv')

raw_datasets = {}
raw_datasets["training_data"] = df_eco
raw_datasets["training_data"], raw_datasets["evaluation_data"] = (
                      sk_ms.train_test_split(raw_datasets["training_data"],
                                               train_size = 1 - 0.3,
                                               test_size = 0.3,
                                               random_state = 123,
                                               shuffle = True) )
datasets = raw_datasets
final_model = sm.logit(formula="Occupancy ~ Temperature + Humidity",
                             data=datasets["training_data"])

final_model = final_model.fit()
data = {'Temperature':[temp], 'Humidity':[hum]}
df = pd.DataFrame(data, columns = ['Temperature', 'Humidity'])
prob = final_model.predict(df)[0]
st.write(f"Probability of being occupied: {prob}")

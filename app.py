import streamlit as st
import pandas as pd
import numpy as np
import sklearn.model_selection as sk_ms
import statsmodels.formula.api as sm
import sklearn.metrics as sk_m
from load_css import local_css
import math
import pickle
local_css("style.css")



"""
# Fedex rate calculator
"""


def dist_between_two_zipcodes(zipA, zipB):
    import mpu
    from uszipcode import SearchEngine
    #for extensive list of zipcodes, set simple_zipcode =False
    search = SearchEngine(simple_zipcode=True)
    distance = 0
    
    try:
        zip1 = search.by_zipcode(zipA)
        lat1 = zip1.lat
        lng1 = zip1.lng

        zip2 = search.by_zipcode(zipB)
        lat2 = zip2.lat
        lng2 = zip2.lng
        distance = mpu.haversine_distance((lat1, lng1), (lat2, lng2)) 

        #converting to miles
        distance = 0.621371 * distance
        return round(distance,1)
    except: 
        return distance  
    

def pricing_zone_finder(zipA, zipB):
    pricing_zone = 'Unknown'
    dist = 0
    int_zipB = int(zipB)
    try:
        if 600 <= int_zipB <= 999:
            pricing_zone = 'P'
            
        elif 96700 <= int_zipB <= 96799:
            pricing_zone = 'H'
            
        elif 96900 <= int_zipB <= 96999:
            pricing_zone = 'Z'
            
        elif 99500 <= int_zipB <= 99999:
            pricing_zone = 'A'
            
        else: 
            dist = dist_between_two_zipcodes(zipA, zipB)

            if dist == 0:
                pricing_zone = 'Unknown'
            elif 0 <dist < 174.4:
                pricing_zone = '2'
            elif 174.4 <= dist < 361.8:
                pricing_zone = '3'
            elif dist < 619.7:
                pricing_zone = '4'
            elif dist < 938.3:
                pricing_zone = '5'
            elif dist < 1333.2:
                pricing_zone = '6'
            elif dist < 1742.4:
                pricing_zone = '7'
            elif dist < 2500.0:
                pricing_zone = '8'
            else: 
                pricing_zone = 'Unknown'
        return pricing_zone
    except:
        return pricing_zone


def cost_finder(pricing_zone, dim_weight, service_type):  
    import pickle
    filename = 'finalized_model.sav'
    final_model = pickle.load(open(filename, 'rb'))

    x_pred = {'shipment_rated_weight_lbs': [0.0],     
            'service_description_Home Delivery': [0.0],     
            'service_description_SmartPost': [0.0],    
            'pricing_zone_ A': [0.0],    
            'pricing_zone_ H': [0.0],    
            'pricing_zone_ P': [0.0],    
            'pricing_zone_ Z': [0.0],    
            'pricing_zone_2.0': [0.0],    
            'pricing_zone_3.0': [0.0],    
            'pricing_zone_4.0': [0.0],    
            'pricing_zone_5.0': [0.0],    
            'pricing_zone_6.0': [0.0],    
            'pricing_zone_7.0': [0.0],    
            'pricing_zone_8.0': [0.0]}
    #assigning pricing zone
    if pricing_zone == 'Unknown':
        return 'n/a'
    elif pricing_zone == 'A':
        x_pred['pricing_zone_ A'] = [1.0]
    elif pricing_zone == 'H': 
        x_pred['pricing_zone_ H'] = [1.0]
    elif pricing_zone == 'P': 
        x_pred['pricing_zone_ P'] = [1.0]
    elif pricing_zone == 'Z': 
        x_pred['pricing_zone_ Z'] = [1.0]
    elif pricing_zone == '2': 
        x_pred['pricing_zone_2.0'] = [1.0]
    elif pricing_zone == '3': 
        x_pred['pricing_zone_3.0'] = [1.0]
    elif pricing_zone == '4': 
        x_pred['pricing_zone_4.0'] = [1.0]
    elif pricing_zone == '5': 
        x_pred['pricing_zone_5.0'] = [1.0]
    elif pricing_zone == '6': 
        x_pred['pricing_zone_6.0'] = [1.0]
    elif pricing_zone == '7': 
        x_pred['pricing_zone_7.0'] = [1.0]
    elif pricing_zone == '8': 
        x_pred['pricing_zone_8.0'] = [1.0]    

    #assigning service_type    
    if service_type == 'smartpost':
        x_pred['service_description_SmartPost'] = [1.0]
    elif service_type == 'home_delivery':
        x_pred['service_description_Home Delivery'] = [1.0]
    #assigning dimensional weight
    x_pred['shipment_rated_weight_lbs'] = [dim_weight]
    
    #changing data from dictionary to dataframe
    df_x_pred = pd.DataFrame(x_pred, columns = ['shipment_rated_weight_lbs',
                                            'service_description_Home Delivery',
                                            'service_description_SmartPost',
                                            'pricing_zone_ A',
                                            'pricing_zone_ H',
                                            'pricing_zone_ P',
                                            'pricing_zone_ Z',
                                            'pricing_zone_2.0',
                                            'pricing_zone_3.0',
                                            'pricing_zone_4.0',
                                            'pricing_zone_5.0',
                                            'pricing_zone_6.0',
                                            'pricing_zone_7.0',
                                            'pricing_zone_8.0'])
    
    
    #making a prediction
    result = round(final_model.predict(df_x_pred)[0],1)
    #return df_x_pred
    return result





st.sidebar.markdown('### Origination & Destination:')
orig_place = st.sidebar.text_input('From:', value = 41048, max_chars = 5)
dest_place = st.sidebar.text_input('To:', value = 10027, max_chars =5 )

st.sidebar.markdown('### Dimensions (inch):')
lenght = st.sidebar.slider("Length", 1.0, 50.0, value = 5.0)
width = st.sidebar.slider("Width", 1.0, 50.0, value =5.0)
height = st.sidebar.slider("Height", 1.0, 50.0, value = 5.0)
st.sidebar.markdown('### Weight:')
weight = st.sidebar.slider("(pounds)", .5, 15.0, value = 5.0)

dist = dist_between_two_zipcodes(orig_place, dest_place)
p_zone = pricing_zone_finder(orig_place, dest_place)
volume = round(lenght) * round(width) * round(height)
dim_weight = max(weight, volume / 130)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#st.write(f'### {loaded_model}')

p_smartpost = cost_finder(p_zone, dim_weight, 'smartpost')
p_home_delivery = cost_finder(p_zone, dim_weight, 'home_delivery')



st.write(f"### Distance: {dist} miles")
st.write(f"### Pricing zone: {p_zone}")
st.write(f"### Shipment weighted rate (lbs): {round(dim_weight,1)}")

"\n"
"\n"
"# Shipment charge:"

st.markdown(f"<h2> SmartPost <span class='highlight red bold'> ${p_smartpost}  </span></h2>", unsafe_allow_html=True)
st.markdown(f"<h2> Home Delivery <span class='highlight blue bold'> ${p_home_delivery}  </span></h2>", unsafe_allow_html=True)

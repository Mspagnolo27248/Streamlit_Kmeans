# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:28:50 2020

@author: Michael
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
#%%
st.write("""
         # ARG Customer Cluster Analysis
#Select the #of Clusters to divide the customer base by. The program will return the most dense grouping shown on the map.

 #        """)
#====================================Create Cache Data=====================================
df = pd.read_csv('Customer_Geo_Gal.csv')
df = df.dropna(how='any') #Remove NaN's
df.sort_values(by='Gallons',ascending=False,inplace=True)
#%%=========================================================================================
st.sidebar.header("User **Input** Parameteres")
minsales = 50000
cluster = 7
classlist = list(df.Class.unique())
#classlist.insert(0,'All')

prodClass = 'All'
#%%===========================================================================================
#Methods
def user_input_features():    
    clusters = st.sidebar.slider('NumberOfClusters',3,10,7)    
    return clusters

def user_input_features_sales():    
    minsales = st.sidebar.slider('Minimum Sales',6000,250000,50000)    
    return minsales

def user_input_boxes():
    prodClass = st.sidebar.selectbox('ProductClass',classlist)
    return prodClass

#===========================================================================================
#Call Methods
cluster = user_input_features()
minsales = user_input_features_sales()
prodClass = user_input_boxes()

df = df[df['Gallons']>minsales]

df = df[df['Class']==prodClass]
#===========================================================================================
# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
X = df.iloc[:,[2,3]].values
kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
df['Group']= y_kmeans
#===========================================================================================
#%%Map Color Outputs
def Colorkey(x):
    if x == 1:
        return 'red'
    elif x==2:
        return 'green'
    elif x==3:
        return 'blue'
    elif x==4:
        return "goldenrod"
    else:
        return 'magenta'
df['Group'] = df['Group']+1
df['Color'] = df['Group'].apply(lambda x: Colorkey(x))     
df.sort_values(by='Group',inplace=True)
#===========================================================================================
#Create Streamlit Outpus
st.dataframe(df)
st.dataframe(df['Group'].value_counts())
st.write('Total:'+str(len(df)))
#===========================================================================================
#%%Create Charts
dataset = df.copy()
fig = px.scatter_mapbox(dataset, lat="Latt", lon="Long", hover_name="CustomerName", hover_data=["Gallons","Group"],
                        color='Color',color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig.show()
st.plotly_chart(fig)


#%%Streamlit Map no extra colors
# midpoint = (np.average(df['Latt']), np.average(df['Long']))
# chart_data = df[['Long','Latt']]
# chart_data.columns = ['lon','lat']
# st.map(chart_data)


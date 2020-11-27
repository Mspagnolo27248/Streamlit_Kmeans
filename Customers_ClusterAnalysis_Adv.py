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
         # Cluster Analysis: Customer Segmentation using K-means Algorithm
         This page shows allows for the interactive segmentation of customers into k groups.
           The k-means algorithm is an unsupervised learning model that groups the customers
           that are closest together.

 #        """)

#====================================Create Cache Data=====================================
df = pd.read_csv('Customer_Geo_Gal.csv')
df = df.dropna(how='any') #Remove NaN's
df.sort_values(by='Gallons',ascending=False,inplace=True)
#%%=========================================================================================
st.sidebar.header("Customize Model Parameters")
minsales = 50000
cluster = 7
classlist = list(df.Class.unique())
#classlist.insert(0,'All')

prodClass = 'All'
#%%===========================================================================================
#Methods
def user_input_features():    
    clusters = st.sidebar.slider('Number Of Clusters',3,10,7)    
    return clusters

def user_input_features_sales():    
    minsales = st.sidebar.slider('Only Customers with Sales Greater Than',6000,250000,50000)    
    return minsales

def user_input_boxes():
    prodClass = st.sidebar.selectbox('Product Class',classlist)
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
        return 'Group1'
    elif x==2:
        return 'Group2'
    elif x==3:
        return 'Group3'
    elif x==4:
        return "Group4"
    elif x==5:
        return "Group5"
    elif x==6:
        return "Group6"
    elif x==7:
        return "Group7"
    elif x==8:
        return "Group8"
    elif x==9:
        return "Group9"
    elif x==10:
        return "Group10"
    else:
        return 'Centers'
df['Group'] = df['Group']+1
df['Color'] = df['Group'].apply(lambda x: Colorkey(x))     
df.sort_values(by='Group',inplace=True)

#===========================================================================================
#Get centorids
# centroids = kmeans.cluster_centers_
# cf = pd.DataFrame({'CustomerName':None,'Class':None,'Long':centroids[:,0],'Latt':centroids[:,1],
#                   'Gallons':0,'Group':None,'Color':'Centers'})
# df = df.append(cf)

#===========================================================================================
#%%Create Charts


dataset = df.copy()

#plot
fig = px.scatter_mapbox(dataset, lat="Latt", lon="Long", hover_name="CustomerName", hover_data=["Gallons","Group"],
                        color='Color', zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#fig.show()
#st.plotly_chart(fig)


#%%Streamlit Map no extra colors
# midpoint = (np.average(df['Latt']), np.average(df['Long']))
# chart_data = df[['Long','Latt']]
# chart_data.columns = ['lon','lat']
# st.map(chart_data)

tbl = df.groupby(['Color']).agg({'Color':'count','Gallons':'sum'})
tbl.columns = ['#OfCustomers','Gallons_Sold']
tbl.sort_values(by='#OfCustomers',inplace=True,ascending=False)

#%%
#Show plotly Express Charts in sypder
#from plotly.offline import plot
#plot(fig)

st.header('Customer Cluster Map: scroll to zoom')
st.plotly_chart(fig)
st.dataframe(tbl.style.format({"Gallons_Sold": "{:,.0f}"}))
st.write('Total Customers :'+str(len(df))+'     '+'Total Gallons: '+f"{df['Gallons'].sum():,d}")
st.subheader('Full Dataset')
st.dataframe(df)


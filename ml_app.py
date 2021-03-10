import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import pickle
import joblib


def run_ml_app():
    st.subheader('Machine Learning')

    ##모델 불러오기

    model = tensorflow.keras.models.load_model('data/check.h5')

    new_data = np.array( [ 0, 38, 90000, 2000, 500000 ] )

    new_data = new_data.reshape(1,-1)

    sc_x = joblib.load('data/sc_X.pkl')
    
    new_data = sc_x.transform(new_data)

    y_pred = model.predict(new_data)

    st.write(y_pred[0][0])

    sc_y = joblib.load('data/sc_y.pkl')

    y_pred_original = sc_y.inverse_transform(y_pred)

    st.write(y_pred_original)